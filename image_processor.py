import os
import cv2
import faiss
import numpy as np
import pandas as pd
import onnxruntime as ort

# STD and MEAN for normalizing input data
MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(-1, 1, 1)
STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(-1, 1, 1)

vit_path = "vit_emb.onnx"
seg_path = "clipseg.onnx"
text_embs = "text_emb.csv"
input_ids = "input_ids.npy"
attention_mask = "attention_mask.npy"
food_base = "food_calories.csv"
"""
col_count = []
with open("data/text_emb.csv", 'r') as temp_f:
    for l in temp_f.readlines():
        col_count.append(l.split(","))
        new_f = col_count[-1][0]
        last_i = 0
        for i in range(1, 10): 
            if '0' not in col_count[-1][i]:
                new_f += f'+{col_count[-1][i]}'
            else:
                last_i = i
                break
        col_count[-1] = [new_f] + col_count[-1][last_i:]
        #print(col_count[-1][0], col_count[-1][1], len(col_count[-1]))
print(col_count[0])
print([len(i) for i in col_count])
df_text_emb = pd.DataFrame(col_count[1:], columns=col_count[0])

df_text_emb.to_csv('text_emb_2.csv', index=False)
text_embs = pd.read_csv('text_emb_2.csv')
print(len(text_embs), len(text_embs.columns))"""


class ImageProcessor:
    """
    Class for processing.
    Embedding model is used to generate embeddings for images. And compare it with base of text.
    Next segment model try to find objects that was found on previous step
    """

    def __init__(
        self,
        image_width_clip=224,
        image_height_clip=224,
        image_width_segclip=352,
        image_height_segclip=352,
        models_folder="models",
        data_folder="data",
    ):
        """
        :param image_width_clip: int, image width for embeder model <br>
        :param image_height_clip: int, height width for embeder model <br>
        :param image_width_segclip: int, image width for segment model <br>
        :param image_height_segclip: int, height width for segment model <br>
        """
        self.text_embs = None
        self.input_ids = None
        self.attention_mask = None
        os.path.join(models_folder, vit_path)

        self.image_width_clip = image_width_clip
        self.image_height_clip = image_height_clip
        self.image_width_segclip = image_width_segclip
        self.image_height_segclip = image_height_segclip
        self.clip = None
        self.segclip = None
        self.text_embs = pd.read_csv(os.path.join(data_folder, text_embs))
        try:
            # read text base
            self.text_embs = pd.read_csv(os.path.join(data_folder, text_embs))
            self.faiss_index = faiss.IndexFlatIP(512)
            self.faiss_index.add(self.text_embs.iloc[:, 1:])

            self.food = pd.read_csv(os.path.join(data_folder, food_base))

            self.input_ids = np.load(os.path.join(data_folder, input_ids))
            self.attention_mask = np.load(os.path.join(data_folder, attention_mask))

            # load models
            self.clip = ort.InferenceSession(
                os.path.join(models_folder, vit_path),
                providers=["CPUExecutionProvider"],
            )
            """clip model"""
            self.segclip = ort.InferenceSession(
                os.path.join(models_folder, seg_path),
                providers=["CPUExecutionProvider"],
            )
            """segclip model"""
            self.clip.disable_fallback()
            self.segclip.disable_fallback()
            self.clip_input_name = self.clip.get_inputs()[0].name
            """clip model input names"""
            self.clip_output_name = self.clip.get_outputs()[0].name
            """clip model output names"""
            # warmup models
            # self.apply_clip_model(np.ones((image_height_clip, image_width_clip, 3),
            #                              dtype=np.uint8))
            # self.apply_segclip_model(np.ones((image_height_clip, image_width_clip, 3),
            #                                  dtype=np.uint8))
        except Exception as exc:
            print(f"Error image retriever init: {exc}")

    def preprocess(self, image, width, height):
        """
        Preprocess image for models<br>
        """
        img_in = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_CUBIC
        )  # resize
        # img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)  # convert to RGB

        img_in = img_in.astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))
        img_in = (img_in - MEAN) / STD

        return img_in

    def apply_clip_model(self, image, preprocess=True):
        """
        Apply clip model to image or images<br>
        """
        model_input = []
        if not isinstance(image, list) or (
            isinstance(image, np.ndarray) and len(image.shape) == 3
        ):
            image = [image]
        for img in image:
            if preprocess:
                img = self.preprocess(
                    img, self.image_width_clip, self.image_height_clip
                )
            img_in = np.ascontiguousarray(img)
            model_input.append(img_in)

        onnx_input_image = {self.clip_input_name: model_input}
        (output,) = self.clip.run(None, onnx_input_image)
        return output

    def apply_segclip_model(self, image, prompts, preprocess=True):
        """
        Apply seg clip model to image or images<br>
        """
        if preprocess:
            img = self.preprocess(
                image, self.image_width_segclip, self.image_height_segclip
            )
        img_in = np.ascontiguousarray(img)

        indxs = list(self.food[self.food.Food.isin(prompts)].index)

        onnx_input = {
            "input_ids": self.input_ids[indxs],
            "pixel_values": [img_in] * len(prompts),
            "attention_mask": self.attention_mask[indxs],
        }
        output = self.segclip.run(None, onnx_input)
        return output[0]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def apply_models(self, image):
        """
        Apply all models end get result of segmentations<br>
        """

        im_emb = self.apply_clip_model(image)
        distances, indices = self.faiss_index.search(im_emb, 1)
        ingredients = self.text_embs.iloc[indices[0], 0].values[0].split("+")
        masks = self.apply_segclip_model(image, ingredients)

        res = {}
        ingr_info = list(self.food[self.food.Food.isin(ingredients)].values)
        for i, ingr in enumerate(ingredients):
            gray_image = (self.sigmoid(masks[i]) * 255).astype(np.uint8)
            (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
            res[ingr] = [int((bw_image / 255).sum())] + list(ingr_info[i][1:])
        return res
