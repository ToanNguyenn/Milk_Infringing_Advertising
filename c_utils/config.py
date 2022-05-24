import torch
import torchvision.transforms as transforms
# Path
train_dir = r'E:\MILK\train_val_test_split\train'
test_dir = r'E:\MILK\train_val_test_split\test'
val_dir = r'E:\MILK\train_val_test_split\val'
# Hyperparameter
seed = 10
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
num_worker = 2
batch_size = 64
pin_memory = True
load_model = False
learning_rate = 1e-4
epochs = 100
num_classes = 157
classes = list(range(157))
# Transform
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
# Class name dictionary
dict_folder = {0: 'Kendamil 4',
 1: 'GerberÂ® Good StartÂ® SoothePro',
 2: 'Nutramigen with Enflora LGG Powder Infant Formula',
 3: 'GerberÂ® Good StartÂ® Gentle Supreme Toddler Drink',
 4: 'Bellamy 8',
 5: 'Bellamy 10',
 6: 'Bellamy 12',
 7: 'CERELAC NUTRIPUFFS BANANA _ ORANGE',
 8: 'CERELAC NUTRIPUFFS BANANA _ STRAWBERRY',
 9: 'CERELAC NutriPuffs Zucchini _ Onion',
 10: 'CERELAC Rice (without milk) or Beras rice',
 11: 'CERELAC Rice _ Mixed Fruits',
 12: 'CERELAC Rice _ Mixed Vegetables',
 13: 'Enfagrow A+ MindPro Step 3',
 14: 'Enfagrow A+ MindPro Step 4',
 15: 'Enfagrow A+ MindPro Step 5',
 16: 'Enfagrow AII MindPro',
 17: 'Enfagrow Pro A+ Stage 4',
 18: 'Enfagrow Pro A+ Stage 5',
 19: 'Enfamil 24 Cal Infant Formula',
 20: 'Enfamil A.R. Infant Formula',
 21: 'Enfamil plant-based',
 22: 'Enfamil Premature Infant Formula',
 23: 'Enfamil Pro A+ Stage 2',
 24: 'Friesland Campina Peak Infant Cereals -Rice',
 25: 'GERBER 1st FOODS Applesauce',
 26: 'Enfamil NeuroPro Sensitive Infant Formula',
 27: 'Enfamil ProSobee Infant Formula',
 28: 'Enfaport Infant Formula',
 29: 'GERBER 1st FOODS Prunes',
 30: 'GERBER Arrowroot Biscuits',
 31: 'GERBER Hearty Bits MultiGrain Banana Apple Strawberry Cereal',
 32: 'GERBER Lil_ Bits Oatmeal Banana Strawberry Cereal',
 33: 'GERBER Lil_ Bits Whole Wheat Apple Blueberry Cereal',
 34: 'Bellamy 7',
 35: 'Bellamy 6 tui',
 36: 'Bellamy 6 hop',
 37: 'Bellamy 4',
 38: 'Bellamy 5',
 39: 'Heinz By Nature 4+',
 40: 'Heinz By Nature 7+',
 41: 'Heinz First steps 7+',
 42: 'Heinz First step 6+',
 43: 'Heinz First steps 4+',
 44: 'Heinz By Nature 10+',
 45: 'GERBER 1st FOODS Carrots',
 46: 'NestlÃ©Â® MOM Maternal Nutritional Supplement 600g',
 47: 'Similac Isomil Stage 1 Soy Protein',
 48: 'Sma Gold Infant Formula',
 49: 'Similac Gain 2â€™-FL Stage 3',
 50: 'GERBER Organic Cereal - Rice',
 51: 'Mamil 4',
 52: 'GERBER Organic Puffs - Cranberry Orange',
 53: 'SMA Gold Toddler Infant Milk',
 54: 'Aptamil Organic Baby Rice Cereal',
 55: 'Similac intelli pro 4',
 56: 'Similac total comfort easy to digest',
 57: 'SimilacÂ® 360 Total CareÂ® Sensitive_',
 58: 'Kendamil 6',
 59: 'GERBER Organic Teether â€“ Blueberry Apple Beet',
 60: 'Instant Milk Powder',
 61: 'SimilacÂ® Soy Isomil',
 62: 'GerberÂ® Good StartÂ® GentlePro',
 63: 'Similac intelli pro 3',
 64: 'MomCare by Similac Prenatal',
 65: 'GERBER Yogurt Melts - Strawberry',
 66: 'GERBER Organic Cereal â€“ Oatmeal Millet Quinoa',
 67: 'GERBER Puffs - Blueberry',
 68: 'GERBER Rice Banana Apple Cereal',
 69: 'GERBER Puffs - Peach',
 70: 'GERBER Oatmeal Peach Apple Cereal',
 71: 'GERBER Puffs - Strawberry Apple',
 72: 'GERBER Organic Teether â€“ Mango Banana Carrot',
 73: 'GERBER Yogurt Melts - Mixed Berries',
 74: 'GerberÂ® Good StartÂ® Extensive HA',
 75: 'GERBER Organic Puffs Apple',
 76: 'GERBER Organic Puffs - Fig Berry',
 77: 'GERBER Puffs - Banana',
 78: 'GERBER Organic Cereal - Oatmeal',
 79: 'Pregestimil Infant Formula',
 80: 'Isomil plus',
 81: 'MAMILÂ® STEP 3',
 82: 'Nutriben 8 Cereals',
 83: 'New and Upgraded Formulation AptamilÂ® Gold+ HA Prosyneo Growing Up Formula',
 84: 'Similac Pro-Sensitive',
 85: 'Pure Blissâ„¢ by Similac',
 86: 'GerberÂ® Good StartÂ® Soy Ready to Feed Infant Formula Nursers',
 87: 'GerberÂ® Good StartÂ® Grow Toddler Probiotic 1+',
 88: 'Similac Pro-Advance',
 89: 'Lactogrow 4',
 90: 'Similac Gain Kid 2â€™-FL Stage 4',
 91: 'Similac For Spit-Up',
 92: 'SimilacÂ® 360 Total CareÂ®_',
 93: 'Kendamil 7 Months Plus Blueberry Porridge',
 94: 'Lactogrow 3',
 95: 'Similac Total Comfort IQ',
 96: 'Pediasure Grow _ Gain Nutrition Shake For Kids',
 97: 'Similac Pro-Total Comfort',
 98: 'NANÂ® Pro',
 99: 'Enfamil NeuroPro Gentlease Infant Formula',
 100: 'SimilacÂ® AdvanceÂ®',
 101: 'SimilacÂ® Alimentum',
 102: 'SimilacÂ® NeoSure',
 103: 'Enfamil Enspire Infant Formula',
 104: 'SMA Pro First Infant Milk',
 105: 'Enfamil Enspire Gentlease Infant Formula',
 106: 'EnfagrowNeuroPro Toddler Nutritional Drink',
 107: 'a2 Smart Nutrition',
 108: 'Aptamil First Infant Milk',
 109: 'a2 Platinum� Premium toddler milk drink',
 110: 'ActivePro Complete Nutrition Formula',
 111: 'Aptamil 4 Growing Up Milk',
 112: 'Enfamil Infant Formula',
 113: 'Enfamil A+ LactoFree, Infant Formula, Lactose-free formula',
 114: 'Enfamil NeuroPro Infant Formula',
 115: 'Enfamil NeuroPro EnfaCare',
 116: 'Enfamil A+ AR, Infant Formula, Stage 1',
 117: 'Enfamil Gentlease Infant Formula',
 118: 'Aptamil Platinum Edition 1',
 119: 'Enfamil A+ Gentlease, Infant Formula',
 120: 'Aptamil Growing Up Milk',
 121: 'Enfamil Simply Organic Infant Formula',
 122: 'Enfamil A+ Soy Care, Infant Formula, Stage 1',
 123: 'Enfamil Premium A2 Infant Formula',
 124: 'Enfamil Reguline Infant Formula',
 125: 'NAN SUPREMEpro 2',
 126: 'Nan Optipro 2',
 127: 'Kendamil Toddler Milk',
 128: 'S-26� GOLD Progress� Stage 3 Growing Up Milk',
 129: 'NAN A2 stage 2',
 130: 'NAN COMFORT 1',
 131: 'Friso� Gold 4 Toddler',
 132: 'Kendamil First Infant Milk',
 133: 'NAN COMFORT 2',
 134: 'Nan Optipro 3',
 135: 'Nan Optipro H.A. 3',
 136: 'NAN SUPREMEpro 1',
 137: 'NAN A.R. Infant Formula for Babies with Regurgitation',
 138: 'Nestle Infant Cereals Honey _ Wheat',
 139: 'Karihome Milk Sweeties Yogurt flavour',
 140: 'PurAmino Junior Unflavore Vanila',
 141: 'Friso� Gold 3 Toddler',
 142: 'NAN� Pro Toddler Drink',
 143: 'Friso GOLD Wheat-Based Milk Cereal',
 144: 'Nestle Nido Kinder 1+ Powdered Milk',
 145: 'NAN A2 stage 1',
 146: 'Nan Optipro 1',
 147: 'Kendamil Follow-On Milk',
 148: 'S-26� GOLD Promise� Stage 4 Growing Up Milk',
 149: 'Similac Gain Plus� Gold 4',
 150: 'a2 Platinum� Premium junior milk drink',
 151: 'a2 Platinum� Premium follow-on formula',
 152: 'a2 Nutrition for mothers',
 153: 'Similac Sensitive',
 154: 'Similac Organic Stage 3',
 155: 'Similac Gain Plus� Gold 3',
 156: 'a2 Platinum� Premium infant formula'}
class_name = dict_folder
