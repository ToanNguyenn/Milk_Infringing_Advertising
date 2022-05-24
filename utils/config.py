import torch
import torchvision.transforms as transforms

config_dir = '/content/drive/MyDrive/AT/Data/Data_Crawl/convnext_config'
data_dir = '/content/drive/MyDrive/AT/Data/Data_Crawl/Data crawl server UK/Images'
model_dir = '/content/drive/MyDrive/AT/Data/Data_Crawl/convnext_config/model'
weight_dir = '/content/drive/MyDrive/AT/Data/Data_Crawl/convnext_config/weight'
working_dir = '/content/drive/MyDrive/AT/Data/Data_Crawl/convnext_config/working'
save_path = ''

train_dir = r'E:\MILK\train_val_test_split\train'
test_dir = r'E:\MILK\train_val_test_split\test'
val_dir = r'E:\MILK\train_val_test_split\val'

seed = 10
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
num_worker = 2
batch_size = 64
pin_memory = True
load_model = False
learning_rate = 1e-4
epochs = 100
num_classes = 50
classes = list(range(50))

img_transforms = transforms.Compose([transforms.Resize([224,224]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ToTensor(),
                                     ])
img_transforms_valid = transforms.Compose([transforms.Resize([224,224]),
                                          #  transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           ])
img_transforms_test = transforms.Compose([transforms.Resize([255,255]),
                                          #  transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          ])

dict_folder = {
    'a2 Smart Nutrition':0,
    'a2 Nutrition for mothers':5,
    'a2 Platinum� Premium follow-on formula':4,
    'a2 Platinum� Premium infant formula':3,
    'a2 Platinum� Premium junior milk drink':2,
    'a2 Platinum� Premium toddler milk drink':1,
    'Kendamil Toddler Milk':6,
    'Kendamil Follow-On Milk':7,
    'Kendamil First Infant Milk':8,
    'ActivePro Complete Nutrition Formula':9,
    'S-26� GOLD Progress� Stage 3 Growing Up Milk':10,
    'S-26� GOLD Promise� Stage 4 Growing Up Milk':11,
    'Similac Gain Plus� Gold 3':14,
    'Similac Gain Plus� Gold 4':13,
    'Similac Sensitive':12,
    'Similac Organic Stage 3':15,
    'PurAmino Junior Unflavore-Vanila':16,
    'Enfamil Gentlease Infant Formula':17,
    'Enfamil Infant Formula':18,
    'Enfamil Premium A2 Infant Formula':19,
    'Enfamil Simply Organic Infant Formula':20,
    'Enfamil A+ AR, Infant Formula, Stage 1':21,
    'Enfamil A+ LactoFree, Infant Formula, Lactose-free formula':22,
    'Enfamil A+ Soy Care, Infant Formula, Stage 1':23,
    'Aptamil Growing Up Milk':24,
    'Aptamil First Infant Milk':25,
    'Aptamil Platinum Edition 1':26,
    'Friso� Gold 4 Toddler':27,
    'Friso� Gold 3 Toddler':28,
    'NAN� Pro Toddler Drink':29,
    'NAN A.R. Infant Formula for Babies with Regurgitation':30,
    'NAN A2 stage 2':31,
    'NAN A2 stage 1':32,
    'NAN COMFORT 1':34,
    'NAN COMFORT 2':33,
    'NAN SUPREMEpro 1':35,
    'Enfamil NeuroPro EnfaCare':36,
    'Enfamil Reguline Infant Formula':37,
    'Enfamil NeuroPro Infant Formula':38,
    'Enfamil A+ Gentlease, Infant Formula':39,
    'Aptamil 4 Growing Up Milk':40,
    'Friso GOLD Wheat-Based Milk Cereal':41,
    'NAN SUPREMEpro 2':42,
    'Nan Optipro H.A. 3':43,
    'Nan Optipro 1':46,
    'Nan Optipro 2':45,
    'Nan Optipro 3':44,
    'Nestle Infant Cereals Honey _ Wheat':47,
    'Nestle Nido Kinder 1+ Powdered Milk':48,
    'Karihome Milk Sweeties Yogurt flavour':49,
}
class_name = {value:key for key, value in dict_folder.items()}