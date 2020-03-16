import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image

class VGObjects:
    def __init__(self,vgjsonfile):

        #self.objs = dict()
        #self.imgs = dict()
        self.w2id = dict()
        self.id2w = dict()
        self.df = pd.read_json(vgjsonfile,orient="split")

        names = Counter(self.df.name)
        for ix,(w,_) in enumerate(names.most_common()):
            self.w2id[w] = ix
            self.id2w[ix] = w

class VGDataset(data.Dataset):

      def __init__(self, root, json, transform=None):

        self.root = root
        self.vg = VGObjects(json)
        self.transform = transform
        self.negpos = 0

        def crop_image(self,img,bb):
            x, y, w, h = np.clip(np.array(bb), 0, np.max(img.shape))
            w = img.shape[1]-x if x+w >= img.shape[1] else w
            h = img.shape[0]-y if y+h >= img.shape[0] else h
            # print 'after', x,y,w,h,
            img_cropped = img[int(y):int(y+h), int(x):int(x+w)]
            return img_cropped

        def get_negatives(self,word,size):
            nobjects = []
            while len(nobjects) < size:
                nrow = self.vg.df.iloc[self.negpos]
                if nrow['name'] != word:

                    bb = nrow['bb']
                    image = Image.open(os.path.join(self.root, path)).convert('RGB')
                    object = self.crop_image(image,bb)
                    if self.transform is not None:
                        object = self.transform(object)
                    nobjects.append(object)
                    self.negpos += 1

                    if self.negpos == len(self.vg.df):
                        self.negpos = 0

            return nobjects

        def __getitem__(self, index):

            row = self.vg.df.iloc[index]
            word = row['name']
            img_id = row['image_id']
            bb = row['bb']

            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            object = self.crop_image(image,bb)
            if self.transform is not None:
                object = self.transform(object)

            wid = self.vg.w2id[word]
            target = torch.Tensor(wid)
            return target, pos_object, neg_objects

        def __len__(self):
            return len(self.vg.df)

def collate_fn(data):
