import torch
import torchvision


class xray_dataset(torch.utils.data.Dataset):
    def __init__(self, df, expand = None, transforms = False, resize = None):
        self.df = df
        self.expand = expand
        self.transforms = transforms
        self.length = len(df)
        self.resize = resize
        
    def __getitem__(self, index):
        
        label, img_path = self.df.iloc[index, 1:]
        label = torch.tensor(label).float() 
        img = torchvision.transforms.functional.to_tensor(Image.open(img_path).resize((self.resize,self.resize), 
                                                                                      resample=PIL.Image.BILINEAR))
        img = img[0:1,:,:]

#         if self.resize!=None:
#             img = torchvision.transforms.functional.resize(img, (self.resize,self.resize))
        
        if self.transforms:
            img = self.transform(img)
        
        if self.expand:
            img = img.expand(3,-1,-1)
            img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        
       
            
        return img, label
    
    def __len__(self):
        return self.length
    
    @staticmethod
    def transform(img):
        if random.random()>0.5:
            img = tr.functional.hflip(img)
        return img
        
        

        
        
    def show_image_from_dataset(self,index):
        print(self[index])
        a = torchvision.transforms.functional.to_pil_image(self[index][0])
        plt.imshow(a)        
        
        
def load_test_validation_df(path, val_split_index):
    train_splits = []
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        df = pd.read_csv(file_path)
        train_splits.append(df) 

    val_splits = []    
    for i in val_split_index:
        val_splits.append(train_splits.pop(i))

    train_df = pd.concat(train_splits, ignore_index = True) 
    val_df = pd.concat(val_splits, ignore_index = True) 
    train_df = train_df.drop(['Unnamed: 0'], axis = 1)
    val_df = val_df.drop(['Unnamed: 0'], axis = 1)
    
    print('train_split_len: ', sum, len(train_df)) 
    print('val_split_len: ', sum, len(val_df))

    return train_df, val_df        