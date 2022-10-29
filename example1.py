import pandas as pd
import numpy as np
import matplotlib
from sklearn import datasets, svm
from statistics import mean
from statistics import median
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv (r'D:\DAI\πτυχιακη\cars\data.csv')
df.head()
df.dtypes
print (df)
print(df.columns)

# statistics
for index in range(16):
    if index == 2 or index == 4 or index == 5 or index == 8 or index == 12 or index == 13 or index == 14 or index == 15:
        sum_col = df.iloc[:,index]
        print ("Mean ", index, " : % s" % (np.mean(sum_col)))
        print ("Max ", index, " : % s" % (max(sum_col)))
        print ("Min ", index, " : % s" % (min(sum_col)))
        print ("Median ", index, " : % s" % (median(sum_col)))
        print ("Standard Deviation ", index, " : % s" % (np.std(sum_col)))

# Dropping irrelevant columns
df = df.drop(['Market Category'], axis=1)
df.head()

df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type":
                        "Transmission", "Driven_Wheels": "Wheels","highway MPG": "MPG-H", "city mpg":
                        "MPG-C", "MSRP": "Price", "Number of Doors": "Doors" })
df.head()

# apply the dtype attribute
result = df.dtypes

print("Output:")
print(result)

# Dropping the duplicate rows
df.shape
duplicate_rows_df = df[df.duplicated()]
print("number of dupicate rows:", duplicate_rows_df.shape)

df.count()
df = df.drop_duplicates()
df.head(5)
df.count()

# Dropping the missing or null values
print(df.isnull().sum())

df = df.dropna()    # Dropping the missing values.
df.count()
print(df.isnull().sum())   # After dropping the values


# Create map for string values
df = df.replace({'Make' : { 'BMW': 0, 'Audi' : 1, 'FIAT': 2, 'Mercedes-Benz': 3, 'Chrysler': 4, 'Nissan': 5, 'Volvo': 6 ,
                      'Mazda': 7, 'Mitsubishi': 8, 'Ferrari': 9, 'Alfa Romeo': 10, 'Toyota': 11, 'Maybach': 12, 
                      'Pontiac': 13, 'McLaren': 14, 'Porsche': 15, 'Saab': 16, 'GMC': 17, 'Hyundai': 18, 
                      'Plymouth': 19, 'Honda': 20, 'Oldsmobile': 21, 'Suzuki': 22, 'Ford': 23, 'Cadillac': 24,
                      'Kia': 25, 'Bentley': 26, 'Chevrolet': 27, 'Dodge': 28, 'Lamborghini': 29, 'Lincoln': 30, 
                      'Subaru': 31, 'Volkswagen': 32, 'Spyker': 33, 'Acura': 34, 'Buick': 35, 'Scion': 36,
                      'Lexus': 37, 'Infiniti': 38, 'Land Rover': 39, 'HUMMER': 40, 'Lotus': 41, 'Genesis': 42,
                      'Maserati': 43, 'Aston Martin': 44, 'Rolls-Royce': 45, 'Bugatti': 46}})
df = df.replace({'Transmission': {'MANUAL': 0, 'AUTOMATIC': 1, 'AUTOMATED_MANUAL': 2, 'UNKNOWN': 3, 'DIRECT_DRIVE': 4}})
df = df.replace({'Wheels' : {'rear wheel drive': 0, 'front wheel drive' : 1, 'all wheel drive': 2, 'four wheel drive': 3}})
df = df.replace({'Engine Fuel Type': {'premium unleaded (required)': 0, 'regular unleaded': 1,
                                      'premium unleaded (recommended)': 2, 'flex-fuel (unleaded/E85)': 3,
                                      'diesel': 4, 'flex-fuel (premium unleaded recommended/E85)': 5,
                                      'electric': 6, 'natural gas': 7, 'flex-fuel (premium unleaded required/E85)': 8}})
df = df.replace({'Vehicle Size': {'Compact': 0, 'Midsize': 1, 'Large': 2 }})
df = df.replace({'Vehicle Style': {'Sedan': 0, 'Coupe': 1, 'Convertible': 2, 'Wagon': 3, '4dr SUV': 4, '2dr Hatchback': 5,
                                   '4dr Hatchback': 6, 'Regular Cab Pickup': 7, 'Extended Cab Pickup': 8,
                                   'Passenger Van': 9, 'Passenger Minivan': 10, 'Cargo Minivan': 11, 'Convertible SUV': 12,
                                   '2dr SUV': 13, 'Cargo Van': 14, 'Crew Cab Pickup': 15}})
df = df.replace({'Model' : { '1 Series M': 0, '1 Series': 1, '100': 2, '124 Spider': 3, '190-Class': 4, '2 Series': 5,
                            '200': 6, '200SX': 7, '240SX': 8, '240': 9, '2': 10, '3 Series Gran Turismo': 11, 
                            '3 Series': 12, '300-Class': 13, '3000GT': 14, '300': 15, '300M': 16, '300ZX': 17, '323': 18,
                            '350-Class': 19, '350Z': 20, '360': 21, '370Z': 22, '3': 23, '4 Series Gran Coupe': 24, 
                            '4 Series': 25, '400-Class': 26, '420-Class': 27, '456M': 28, '458 Italia': 29, '4C': 30, 
                            '4Runner': 31, '5 Series Gran Turismo': 32, '5 Series': 33, '500-Class': 34, '500': 35, 
                            '500L': 36, '500X': 37, '550': 38, '560-Class': 39, '570S': 40, '575M': 41, '57': 42, 
                            '599': 43, '5': 44, '6 Series Gran Coupe': 45, '6 Series': 46, '600-Class': 47, '6000': 48,
                            '612 Scaglietti': 49, '626' : 50, '62': 51, '650S Coupe': 52, '650S Spider': 53, '6': 54, 
                            '7 Series': 55, '718 Cayman': 56, '740': 57, '760': 58, '780': 59, '8 Series': 60, '80': 61,
                            '850': 62, '86': 63, '9-2X': 64, '9-3 Griffin': 65, '9-3': 66, '9-4X': 67, '9-5': 68, 
                            '9-7X': 69, '9000': 70, '900': 71, '90': 72, '911': 73, '928': 74, '929': 75, '940': 76, 
                            '944': 77, '960': 78, '968': 79, 'A3': 80, 'A4 allroad': 81, 'A4': 82, 'A5': 83, 'A6': 84,
                            'A7': 85, 'A8': 86, 'Acadia Limited': 87, 'Acadia': 88, 'Accent': 89, 'Acclaim': 90, 
                            'Accord Crosstour': 91, 'Accord Hybrid': 92, 'Accord Plug-In Hybrid': 93, 'Accord': 94, 
                            'Achieva': 95, 'ActiveHybrid 5': 96, 'ActiveHybrid 7': 97, 'ActiveHybrid X6': 98, 
                            'Aerio': 99, 'Aerostar': 100, 'Alero': 101, 'Allante': 102, 'allroad quattro': 103, 
                            'allroad': 104, 'ALPINA B6 Gran Coupe': 105, 'ALPINA B7': 106, 'Alpina': 107, 
                            'Altima Hybrid': 108, 'Altima': 109, 'Amanti': 110, 'AMG GT': 111, 'Armada': 112, 
                            'Arnage': 113, 'Aspen': 114, 'Aspire': 115, 'Astro Cargo': 116, 'Astro': 117, 
                            'ATS Coupe': 118, 'ATS-V': 119, 'ATS': 120, 'Aurora': 121, 'Avalanche': 122, 
                            'Avalon Hybrid': 123, 'Avalon': 124, 'Avenger': 125, 'Aventador': 126, 'Aveo': 127,
                            'Aviator': 128, 'Axxess': 129, 'Azera': 130, 'Aztek': 131, 'Azure T': 132, 'Azure': 133,
                            'B-Class Electric Drive': 134, 'B-Series Pickup': 135, 'B-Series Truck': 136, 'B-Series': 137,
                            'B9 Tribeca': 138, 'Baja': 139, 'Beetle Convertible': 140, 'Beetle': 141, 'Beretta': 142,
                            'Black Diamond Avalanche': 143, 'Blackwood': 144, 'Blazer': 145, 'Bonneville': 146,
                            'Borrego': 147, 'Boxster': 148, 'Bravada': 149, 'Breeze': 150, 'Bronco II': 151, 'Bronco': 152,
                            'Brooklands': 153, 'Brougham': 154, 'BRZ': 155, 'C-Class': 156, 'C-Max Hybrid': 157,
                            'C30': 158, 'C36 AMG': 159, 'C43 AMG': 160, 'C70': 161, 'C8': 162, 'Cabriolet': 163,
                            'Cabrio': 164, 'Cadenza': 165, 'Caliber': 166, 'California T': 167, 'California': 168,
                            'Camaro': 169, 'Camry Hybrid': 170, 'Camry Solara': 171, 'Camry': 172, 'Canyon': 173,
                            'Caprice': 174, 'Captiva Sport': 175, 'Caravan': 176, 'Carrera GT': 177, 'Cascada': 178,
                            'Catera': 179, 'Cavalier': 180, 'Cayenne': 181, 'Cayman S': 182, 'Cayman': 183, 'CC': 184,
                            'Celebrity': 185, 'Celica': 186, 'Century': 187, 'Challenger': 188, 'Charger': 189, 
                            'Chevy Van': 190, 'Ciera': 191, 'Cirrus': 192, 'City Express': 193, 'Civic CRX': 194,
                            'Civic del Sol': 195, 'Civic': 196, 'C/K 1500 Series': 197, 'C/K 2500 Series': 198,
                            'CL-Class': 199, 'CLA-Class': 200, 'CL': 201, 'Classic': 202, 'CLK-Class': 203,
                            'CLS-Class': 204, 'Cobalt': 205, 'Colorado': 206, 'Colt': 207, 'Concorde': 208,
                            'Continental Flying Spur Speed': 209, 'Continental Flying Spur': 210,
                            'Continental GT Speed Convertible': 211, 'Continental GT Speed': 212,
                            'Continental GT3-R': 213, 'Continental GT': 214, 'Continental GTC Speed': 215,
                            'Continental GTC': 216, 'Continental Supersports Convertible': 217,
                            'Continental Supersports': 218, 'Continental': 219, 'Contour SVT': 220, 'Contour': 221,
                            'Corniche': 222, 'Corolla iM': 223, 'Corolla': 224, 'Corrado': 225, 'Corsica': 226,
                            'Corvette Stingray': 227, 'Corvette': 228, 'Coupe': 229, 'CR-V': 300, 'CR-Z': 301,
                            'Cressida': 302, 'Crossfire': 303, 'Crosstour': 304, 'Crosstrek': 305, 'Crown Victoria': 306,
                            'Cruze Limited': 307, 'Cruze': 308, 'CT 200h': 309, 'CT6': 310, 'CTS Coupe': 311,
                            'CTS-V Coupe': 312, 'CTS-V Wagon': 313, 'CTS-V': 314, 'CTS Wagon': 315, 'CTS': 316,
                            'Cube': 317, 'Custom Cruiser': 318, 'Cutlass Calais': 319, 'Cutlass Ciera': 320,
                            'Cutlass Supreme': 321, 'Cutlass': 322, 'CX-3': 323, 'CX-5': 324, 'CX-7': 325, 'CX-9': 326,
                            'Dakota': 327, 'Dart': 328, 'Dawn': 329, 'Daytona': 330, 'DB7': 331, 'DB9 GT': 332, 'DB9': 333,
                            'DBS': 334, 'Defender': 335, 'DeVille': 336, 'Diablo': 337, 'Diamante': 338,
                            'Discovery Series II': 339, 'Discovery Sport': 340, 'Discovery': 341, 'DTS': 342,
                            'Durango': 343, 'Dynasty': 344, 'E-150': 345, 'E-250': 346, 'E-Class': 347, 'E-Series Van': 348,
                            'E-Series Wagon': 349, 'E55 AMG': 350, 'ECHO': 351, 'Eclipse Spyder': 352, 'Eclipse': 353,
                            'Edge': 354, 'Eighty-Eight Royale': 355, 'Eighty-Eight': 356, 'Elantra Coupe': 357,
                            'Elantra GT': 358, 'Elantra Touring': 359, 'Elantra': 360, 'Eldorado': 361, 'Electra': 362,
                            'Element': 363, 'Elise': 364, 'Enclave': 365, 'Encore': 366, 'Endeavor': 367, 'Entourage':368,
                            'Envision': 369, 'Envoy XL': 370, 'Envoy XUV': 371, 'Envoy': 372, 'Enzo': 373, 'Eos': 374,
                            'Equator': 375, 'Equinox': 376, 'Equus': 377, 'ES 250': 378, 'ES 300h': 379, 'ES 300': 380,
                            'ES 330': 381, 'ES 350': 382, 'Escalade ESV': 383, 'Escalade EXT': 384, 'Escalade Hybrid': 385,
                            'Escalade': 386, 'Escape Hybrid': 387, 'Escape': 388, 'Escort': 389, 'Esprit': 390,
                            'Estate Wagon': 391, 'Esteem': 392, 'EuroVan': 393, 'Evora 400': 394, 'Evora': 395, 'EX35': 396,
                            'Excel': 397, 'Exige': 398, 'EX': 399, 'Expedition': 400, 'Explorer Sport Trac': 401,
                            'Explorer Sport': 402, 'Explorer': 403, 'Expo': 404, 'Express Cargo': 405, 'Express': 406,
                            'F-150 Heritage': 407, 'F-150 SVT Lightning': 408, 'F-150': 409, 'F-250': 410, 
                            'F12 Berlinetta': 411, 'F430': 412, 'Festiva': 413, 'FF': 414, 'Fiesta': 415, 'Firebird': 416,
                            'Fit': 417, 'Five Hundred': 418, 'FJ Cruiser': 419, 'Fleetwood': 420, 'Flex': 421,
                            'Flying Spur': 422, 'Focus RS': 423, 'Focus ST': 424, 'Focus': 425, 'Forenza': 426,
                            'Forester': 427, 'Forte': 428, 'Fox': 429, 'FR-S': 430, 'Freelander': 431, 'Freestar': 432,
                            'Freestyle': 433, 'Frontier': 434, 'Fusion Hybrid': 435, 'Fusion': 436, 'FX35': 437, 'FX45': 438,
                            'FX50': 439, 'FX': 450, 'G-Class': 451, 'G Convertible': 452, 'G Coupe': 453, 'G Sedan': 454, 
                            'G20': 455, 'G35': 456, 'G37 Convertible': 457, 'G37 Coupe': 458, 'G37 Sedan': 459, 'G37': 460,
                            'G3': 461, 'G5': 462, 'G6': 463, 'G80': 464, 'G8': 465, 'Galant': 466, 'Gallardo': 467,
                            'Genesis Coupe': 468, 'Genesis': 469, 'Ghibli': 470, 'Ghost Series II': 471, 'Ghost': 472,
                            'GL-Class': 473, 'GLA-Class': 474, 'GLC-Class': 475, 'GLE-Class Coupe': 476, 'GLE-Class': 477,
                            'GLI': 478, 'GLK-Class': 479, 'GLS-Class': 480, 'Golf Alltrack': 481, 'Golf GTI': 482,
                            'Golf R': 483, 'Golf SportWagen': 484, 'Golf': 485, 'Grand Am': 486, 'Grand Caravan': 487,
                            'Grand Prix': 488, 'Grand Vitara': 489, 'Grand Voyager': 490, 'GranSport': 491,
                            'GranTurismo Convertible': 492, 'GranTurismo': 493, 'GS 200t': 493, 'GS 300': 494, 'GS 350': 495,
                            'GS 400': 496, 'GS 430': 497, 'GS 450h': 498, 'GS 460': 499, 'GS F': 500, 'GT-R': 501,
                            'GT': 502, 'GTI': 503, 'GTO': 504, 'GX 460': 505, 'GX 470': 506, 'H3': 507, 'H3T': 507,
                            'HHR': 508, 'Highlander Hybrid': 509, 'Highlander': 510, 'Horizon': 511, 'HR-V': 512,
                            'HS 250h': 513, 'Huracan': 514, 'I30': 515, 'I35': 516, 'i3': 517, 'iA': 518, 'ILX Hybrid': 519,
                            'ILX': 520, 'Impala Limited': 521, 'Impala': 522, 'Imperial': 523, 'Impreza WRX': 524, 
                            'Impreza': 525, 'iM': 526, 'Insight': 527, 'Integra': 528, 'Intrepid': 529, 'Intrigue': 530,
                            'iQ': 531, 'IS 200t': 532, 'IS 250 C': 533, 'IS 250': 534, 'IS 300': 535, 'IS 350 C': 536,
                            'IS 350': 537, 'IS F': 538, 'J30': 539, 'Jetta GLI': 540, 'Jetta Hybrid': 541,
                            'Jetta SportWagen': 542, 'Jetta': 543, 'Jimmy': 544, 'Journey': 545, 'Juke': 546, 'Justy': 547,
                            'JX': 547, 'K900': 548, 'Kizashi': 549, 'LaCrosse': 550, 'Lancer Evolution': 551,
                            'Lancer Sportback': 552, 'Lancer': 553, 'Land Cruiser': 554, 'Landaulet': 555, 'Laser': 556,
                            'Le Baron': 557, 'Le Mans': 558, 'Legacy': 559, 'Legend': 560, 'LeSabre': 561, 'Levante': 562,
                            'LFA': 563, 'LHS': 564, 'Loyale': 565, 'LR2': 566, 'LR3': 567, 'LR4': 568, 'LS 400': 569,
                            'LS 430': 570, 'LS 460': 571, 'LS 600h L': 572, 'LS': 573, 'LSS': 574, 'LTD Crown Victoria': 575,
                            'Lucerne': 576, 'Lumina Minivan': 577, 'Lumina': 578, 'LX 450': 579, 'LX 470': 580, 
                            'LX 570': 581, 'M-Class': 582, 'M2': 583, 'M30': 584, 'M35': 585, 'M37': 586, 'M3': 587,
                            'M4 GTS': 588, 'M45': 589, 'M4': 590, 'M56': 591, 'M5': 592, 'M6 Gran Coupe': 593, 'M6': 594,
                            'Macan': 595, 'Magnum': 596, 'Malibu Classic': 597, 'Malibu Hybrid': 598, 'Malibu Limited': 599,
                            'Malibu Maxx': 600, 'Malibu': 601, 'Mark LT': 602, 'Mark VIII': 603, 'Mark VII': 604,
                            'Matrix': 605, 'Maxima': 606, 'Maybach': 607, 'Mazdaspeed 3': 608, 'Mazdaspeed 6': 609,
                            'Mazdaspeed MX-5 Miata': 610, 'Mazdaspeed Protege': 611, 'M': 612, 'MDX': 613, 'Metris': 614,
                            'Metro': 615, 'Mighty Max Pickup': 616, 'Millenia': 617, 'Mirage G4': 618, 'Mirage': 619,
                            'MKC': 620, 'MKS': 621, 'MKT': 622, 'MKX': 623, 'MKZ Hybrid': 624, 'MKZ': 625, 'ML55 AMG': 626,
                            'Monaco': 627, 'Montana SV6': 628, 'Montana': 629, 'Monte Carlo': 630, 'Montero Sport': 631,
                            'Montero': 632, 'MP4-12C': 633, 'MPV': 634, 'MR2 Spyder': 635, 'MR2': 636, 'Mulsanne': 637,
                            'Murano CrossCabriolet': 638, 'Murano': 639, 'Murcielago': 640, 'Mustang SVT Cobra': 641,
                            'Mustang': 642, 'MX-3': 643, 'MX-5 Miata': 644, 'MX-6': 645, 'Navajo': 646, 'Navigator': 647,
                            'Neon': 648, 'New Beetle': 649, 'New Yorker': 650, 'Ninety-Eight': 651, 'Nitro': 652, 'NSX': 653,
                            'NV200': 654, 'NX 200t': 655, 'NX 300h': 656, 'NX': 657, 'Odyssey': 658, 'Omni': 659,
                            'Optima Hybrid': 660, 'Optima': 661, 'Outback': 662, 'Outlander Sport': 663, 'Outlander': 664,
                            'Pacifica': 665, 'Panamera': 666, 'Park Avenue': 667, 'Park Ward': 668, 'Paseo': 669, 
                            'Passat': 670, 'Passport': 671, 'Pathfinder': 672, 'Phaeton': 673, 'Phantom Coupe': 674,
                            'Phantom Drophead Coupe': 675, 'Phantom': 676, 'Pickup': 677, 'Pilot': 678, 'Precis': 679,
                            'Prelude': 680, 'Previa': 681, 'Prius c': 682, 'Prius Prime': 683, 'Prius v': 684, 'Prius': 685,
                            'Prizm': 686, 'Probe': 687, 'Protege5': 688, 'Protege': 689, 'Prowler': 690, 'PT Cruiser': 691,
                            'Pulsar': 692, 'Q3': 693, 'Q40': 694, 'Q45': 695, 'Q50': 696, 'Q5': 697, 'Q60 Convertible': 698,
                            'Q60 Coupe': 699, 'Q70': 700, 'Q7': 701, 'Quattroporte': 702, 'Quest': 703, 'QX4': 704,
                            'QX50': 705, 'QX56': 706, 'QX60': 707, 'QX70': 708, 'QX80': 709, 'QX': 710, 'R-Class': 711,
                            'R32': 712, 'R8': 713, 'Rabbit': 714, 'Raider': 715, 'Rainier': 716, 'Rally Wagon': 717,
                            'RAM 150': 718, 'RAM 250': 719, 'Ram 50 Pickup': 720, 'Ram Cargo': 721, 'Ram Pickup 1500': 722,
                            'Ram Van': 723, 'Ram Wagon': 724, 'Ramcharger': 725, 'Range Rover Evoque': 726, 
                            'Range Rover Sport': 727, 'Range Rover': 728, 'Ranger': 729, 'Rapide S': 730, 'Rapide': 731,
                            'RAV4 Hybrid': 732, 'RAV4': 733, 'RC 200t': 734, 'RC 300': 735, 'RC 350': 736, 'RC F': 737,
                            'RDX': 738, 'Reatta': 739, 'Regal': 740, 'Regency': 741, 'Rendezvous': 742, 'Reno': 743,
                            'Reventon': 744, 'Ridgeline': 745, 'Rio': 746, 'Riviera': 747, 'RL': 748, 'RLX': 749,
                            'Roadmaster': 750, 'Rogue Select': 751, 'Rogue': 752, 'Rondo': 753, 'Routan': 754, 'RS 4': 755,
                            'RS 5': 756, 'RS 6': 757, 'RS 7': 758, 'RSX': 759, 'RX 300': 760, 'RX 330': 761, 'RX 350': 762,
                            'RX 400h': 763, 'RX 450h': 764, 'S-10 Blazer': 765, 'S-10': 766, 'S-15 Jimmy': 767, 'S-15': 768,
                            'S-Class': 769, 'S2000': 770, 'S3': 771, 'S40': 772, 'S4': 773, 'S5': 774,
                            'S60 Cross Country': 775, 'S60': 776, 'S6': 777, 'S70': 778, 'S7': 779, 'S80': 780, 'S8': 781,
                            'S90': 782, 'Safari Cargo': 783, 'Safari': 784, 'Samurai': 785, 'Santa Fe Sport': 786,
                            'Santa Fe': 787, 'Savana Cargo': 788, 'Savana': 789, 'SC 300': 790, 'SC 400': 791,
                            'SC 430': 792, 'Scoupe': 793, 'Sebring':794, 'Sedona': 795, 'Sentra': 796, 'Sephia': 797,
                            'Sequoia': 798, 'Seville': 799, 'Shadow': 800, 'Shelby GT350': 801, 'Shelby GT500': 802,
                            'Sidekick': 803, 'Sienna': 804, 'Sierra 1500 Classic': 805, 'Sierra 1500 Hybrid': 806,
                            'Sierra 1500': 807, 'Sierra 1500HD': 808, 'Sierra C3': 809, 'Sierra Classic 1500': 810,
                            'Sigma': 811, 'Silhouette': 812, 'Silver Seraph': 813, 'Silverado 1500 Classic': 814,
                            'Silverado 1500 Hybrid': 815, 'Silverado 1500': 816, 'Sixty Special': 817, 'Skylark': 818,
                            'SL-Class': 819, 'SLC-Class': 820, 'SLK-Class': 821, 'SLR McLaren': 822, 
                            'SLS AMG GT Final Edition': 823, 'SLS AMG GT': 824, 'SLS AMG': 825, 'SLX': 826, 'Solstice': 827,
                            'Sonata Hybrid': 828, 'Sonata': 829, 'Sonic': 830, 'Sonoma': 831, 'Sorento': 831, 'Soul': 832,
                            'Spark EV': 833, 'Spark': 834, 'Spectra': 835, 'Spirit': 836, 'Sportage': 837, 'Sportvan': 838,
                            'Spyder': 839, 'SQ5': 840, 'SRT Viper': 841, 'SRX': 842, 'SS': 843, 'SSR': 844, 'Stanza': 845,
                            'Stealth': 846, 'Stratus': 847, 'STS-V': 848, 'STS': 849, 'Suburban': 850, 'Sunbird': 851,
                            'Sundance': 852, 'Sunfire': 853, 'Superamerica': 854, 'Supersports Convertible ISR': 855,
                            'Supra': 856, 'SVX': 857, 'Swift': 858, 'SX4': 859, 'Syclone': 860, 'T100': 861, 'Tacoma': 862,
                            'Tahoe Hybrid': 863, 'Tahoe Limited/Z71': 864, 'Tahoe': 865, 'Taurus X': 866, 'Taurus': 867,
                            'TC': 868, 'tC': 869, 'Tempo': 870, 'Tercel': 871, 'Terrain': 872, 'Terraza': 873,
                            'Thunderbird': 874, 'Tiburon': 875, 'Tiguan': 876, 'Titan': 877, 'TL': 878, 'TLX': 879,
                            'Toronado': 880, 'Torrent': 881, 'Touareg 2': 882, 'Touareg': 883, 'Town and Country': 884,
                            'Town Car': 885, 'Tracker': 886, 'TrailBlazer EXT': 887, 'TrailBlazer': 888, 'Trans Sport': 889,
                            'Transit Connect': 890, 'Transit Wagon': 891, 'Traverse': 892, 'Trax': 893, 'Tribeca': 894,
                            'Tribute Hybrid': 895, 'Tribute': 896, 'Truck': 897, 'TSX Sport Wagon': 898, 'TSX': 899,
                            'TT RS': 900, 'TT': 901, 'TTS': 902, 'Tucson': 903, 'Tundra': 904, 'Typhoon': 905,
                            'Uplander': 906, 'V12 Vanquish': 907, 'V12 Vantage S': 908, 'V12 Vantage': 909, 'V40': 910,
                            'V50': 911, 'V60 Cross Country': 912, 'V60': 913, 'V70': 914, 'V8 Vantage': 915, 'V8': 916,
                            'V90': 917, 'Vanagon': 918, 'Vandura': 919, 'Van': 920, 'Vanquish': 921, 'Vanwagon': 922,
                            'Veloster': 923, 'Venture': 924, 'Venza': 925, 'Veracruz': 926, 'Verano': 927, 'Verona': 928,
                            'Versa Note': 929, 'Versa': 930, 'Veyron 16.4': 931, 'Vibe': 932, 'Vigor': 933, 'Viper': 924,
                            'Virage': 935, 'Vitara': 936, 'Voyager': 937, 'Windstar Cargo': 938, 'Windstar': 939,
                            'Wraith': 940, 'WRX': 941, 'X-90': 942, 'X1': 943, 'X3': 944, 'X4': 945, 'X5 M': 946, 'X5': 947,
                            'X6 M': 948, 'X6': 949, 'xA': 950, 'xB': 951, 'XC60': 952, 'XC70': 953, 'XC90': 954, 'XC': 955,
                            'xD': 956, 'XG300': 957, 'XG350': 958, 'XL-7': 959, 'XL7': 960, 'XLR-V': 961, 'XLR': 962,
                            'XT5': 963, 'Xterra': 964, 'XTS': 965, 'XT': 966, 'XV Crosstrek': 967, 'Yaris iA': 968,
                            'Yaris': 969, 'Yukon Denali': 970, 'Yukon Hybrid': 971, 'Yukon XL': 972, 'Yukon': 973,
                            'Z3': 974, 'Z4 M': 975, 'Z4': 976, 'Z8': 977, 'ZDX': 978, 'Zephyr': 979 }})




# Prediction Model Building
X = df[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C', 'Doors', 'Make', 'Model', 'Transmission', 'Wheels', 
        'Engine Fuel Type', 'Vehicle Size', 'Popularity', 'Vehicle Style']]
y = df.Price
#df["Make"] = df["Make"].astype("float64")
#print(df['Make'].dtypes)


num = int(len(df)*0.8)
train = df[:num]
test = df[num:]
print ("Data:",len(df),", Train:",len(train),", Test:",len(test))

# Create a better train-test split
df1 = df.query("Price < 25001");
percentage1 = (df1.shape[0] / df.shape[0]) * 100
print("df1:", percentage1)
X1 = df1[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C', 'Doors', 'Make', 'Transmission', 'Wheels', 'Model', 
          'Vehicle Size', 'Engine Fuel Type', 'Popularity', 'Vehicle Style' ]]
y1 = df1.Price

df2 = df.query("Price > 25000 & Price < 50001");
percentage2 = (df2.shape[0] / df.shape[0]) * 100
print("df2:", percentage2)
X2 = df2[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C', 'Doors', 'Make', 'Transmission', 'Wheels', 'Model', 
          'Vehicle Size', 'Engine Fuel Type', 'Popularity', 'Vehicle Style' ]]
y2 = df2.Price

df3 = df.query("Price > 50000 & Price < 100001");
percentage3 = (df3.shape[0] / df.shape[0]) * 100
print("df3:", percentage3)
X3 = df3[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C', 'Doors', 'Make', 'Transmission', 'Wheels', 'Model', 
          'Vehicle Size', 'Engine Fuel Type', 'Popularity', 'Vehicle Style']]
y3 = df3.Price

df4 = df.query("Price > 100000");
percentage4 = (df4.shape[0] / df.shape[0]) * 100
print("df4:", percentage4)
X4 = df4[['Year', 'HP', 'Cylinders', 'MPG-H', 'MPG-C', 'Doors', 'Make', 'Transmission', 'Wheels', 'Model',
          'Vehicle Size', 'Engine Fuel Type', 'Popularity', 'Vehicle Style']]
y4 = df4.Price


# Split DataSet in test and train
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 8)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 8)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 8)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 8)
X_train = pd.concat([X1_train, X2_train, X3_train, X4_train], axis = 0)
X_test = pd.concat([X1_test, X2_test, X3_test, X4_test], axis = 0)
y_train = pd.concat([y1_train, y2_train, y3_train, y4_train], axis = 0)
y_test = pd.concat([y1_test, y2_test, y3_test, y4_test], axis = 0)
#X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2, random_state = 8)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# creating an object of LinearRegression class
lr = LinearRegression()
# fitting the training data
model = lr.fit(X_train,y_train)

# make predictions
y_pred = lr.predict(X_test)
print(y_pred)

## The line / model
plt.scatter(y_test, y_pred)
plt.title("Cars Data Set")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


## Score
print ("Score : ", model.score(X_test, y_test))
print("Training set score:", lr.score(X_train, y_train))
print("Test set score:", lr.score(X_test, y_test))

# evaluate predictions
coefficients = lr.coef_
intercept = lr.intercept_

y_pred = lr.predict(X_test)
print("Coefficients: \n", coefficients)
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)
#print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


#Generate the confusion matrix
#cf_matrix = confusion_matrix(y_test, y_pred)

#print(cf_matrix)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#from sklearn import tree
#tree.plot_tree(clf_en.fit(X_train, y_train))

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred_en)
#print('Confusion matrix\n\n', cm)

#np.set_printoptions(precision=2)




