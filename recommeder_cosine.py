#@author Arpan Sarkar
#START PGP MD5
#c5dbab9480173ef71df02e9b7721aa65277588f102243d14b1ed1ed0c38a09ef
#END PGP MD5

# Import Pandas
import pandas as pd
import os
import time
# Import TFID vectorizer for word processing
from sklearn.feature_extraction.text import TfidfVectorizer

os.system("cls")#clear screen

def load_data():
    # Load Movies Metadata
    print("STARTING DATA IMPORT AND PRE-PROCESSING")
    #time.sleep(5)
    metadata = pd.read_csv('prod.csv', low_memory=False)
    metadata = metadata.apply(lambda x: x.str.lower() if x.dtype == "object" else x)  
    # Print head
    print(metadata.head())
    print("************************************************************************************************************")

    # View Columns
    print(metadata.columns.values)
    print("************************************************************************************************************")

    # Extract Columns
    metadata = metadata[['prod_name','prod_cat','prod_cst','prod_weight','prod_features']]
    #time.sleep(5)
    # view extraction
    print(metadata.head())
    print("************************************************************************************************************")

    # handle inaccuracy
    metadata = metadata.fillna('')
    #time.sleep(5)
    print(metadata['prod_features'].head())
    print("************************************************************************************************************")

    # Call TFIDF 
    tfidf = TfidfVectorizer(stop_words='english')

    #Fit data to tfidf
    tfidf_matrix = tfidf.fit_transform(metadata['prod_features'])
    #time.sleep(5)
    #Check
    print(tfidf_matrix.shape)
    print("************************************************************************************************************")
    print(tfidf.get_feature_names())
    print("************************************************************************************************************")

    #Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel

    #Calculate cosine similarity matrix mat(A) * mat(A) = mat(a)^2
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #time.sleep(5)
    #Check cosine matrix
    print(cosine_sim.shape)
    print("************************************************************************************************************")
    print(cosine_sim[1])
    print("************************************************************************************************************")
    #Map to product names
    indices = pd.Series(metadata.index, index=metadata['prod_name']).drop_duplicates()
    #time.sleep(5)
    print(indices[:5])
    print("************************************************************************************************************")
    return metadata,cosine_sim

#@author Arpan Sarkar
#START PGP MD5
#c5dbab9480173ef71df02e9b7721aa65277588f102243d14b1ed1ed0c38a09ef
#END PGP MD5


def get_recommendations(name, cosine_sim, metadata, n):

    #time.sleep(6)
    indices = pd.Series(metadata.index, index=metadata['prod_name']).drop_duplicates()
    idx = indices[name]

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar products
    sim_scores = sim_scores[1:n+1]

    # Get the product indices
    prod_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar product
    return metadata['prod_name'].iloc[prod_indices]


#Basic recommendation based on product keywords
def basic_recc(x, n):
    metadata, cosine_sim = load_data()
    x = str(x)
    print("CALCULATING NEIGHBOUR SIMILARITIES USING FEATURES")
    result = get_recommendations(x,cosine_sim, metadata, n)
    print("\n\n\n\n\n\n\n")
    print("******************RECOMMENDATIONS**********************")
    print(result)
    print("*******************************************************")
    print("\n\n\n\n\n\n\n")
    return result

def adv_recc(x, n):
    #x = str(x)
    metadata, cosine_sim = calc()
    metadata = metadata.reset_index()
    #Map to product names
    print("CALCULATING NEIGHBOUR SIMILARITIES USING FEATURES AND METADATA")
    indices = pd.Series(metadata.index, index=metadata['prod_name'])
    print("\n\n\n\n\n\n\n")
    print("******************RECOMMENDATIONS**********************")
    print(get_recommendations(x, cosine_sim, metadata, n))
    print("*******************************************************")
    print("\n\n\n\n\n\n\n")
    return get_recommendations(x, cosine_sim, metadata, n)

#@author Arpan Sarkar
#START PGP MD5
#c5dbab9480173ef71df02e9b7721aa65277588f102243d14b1ed1ed0c38a09ef
#END PGP MD5


##PLot results
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import umap.plot


def calc():

    metadata , cosine = load_data()
     #Clean data like cost , weight, description
    def clean_data(x):
        return str.lower(str(x).replace(" ", "")) #remove space

    #declare features to be cleaned
    features = ['prod_cat','prod_cst','prod_weight','prod_features']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data) #apply clean method

    print(metadata.head())
    print("************************************************************************************************************")

    #Join all features with space as delimiter to create stringified description
    def create_comb(x):
        return ' '+ x['prod_cat'] + ' ' +x['prod_cst'] + ' ' + x['prod_weight'] + ' '+x['prod_features'].replace(","," ")

    metadata['comb'] = metadata.apply(create_comb, axis=1)
    
    print(metadata[['comb']].head(2)) #check
    print("************************************************************************************************************")

    # Import CountVectorizer and create the count matrix
    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer(stop_words='english')
    #fit stringified description
    count_matrix = count.fit_transform(metadata['comb'])

    print(count_matrix.shape)
    print("************************************************************************************************************")

    from sklearn.metrics.pairwise import cosine_similarity

    #calculate mat(A) * mat(A) = mat(A)^2
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    return metadata, cosine_sim2

#@author Arpan Sarkar
#START PGP MD5
#c5dbab9480173ef71df02e9b7721aa65277588f102243d14b1ed1ed0c38a09ef
#END PGP MD5

def show_figure():

    metadata, cosine_sim2 = calc()
    sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

    data = cosine_sim2
    plt.plot(cosine_sim2, '-.')
    plt.title("Relativity")
    plt.xlabel("Product")
    plt.ylabel("Cosine Similarity")
    #Set number of neighbors to be fetched
    mapper = umap.UMAP(n_neighbors=15, metric='cosine').fit(data)
    fig = umap.plot.connectivity(mapper, show_points=True)
    figl = umap.plot.points(mapper)
    plt.show()


#BETA FUNCTION  NOT STABLE
def collaborative(n):
    coll_data = []
    print("Enter 5 previous purchase histories (product name)")
    for i in range(5):
        print("PRODUCT "+str(i+1))
        tr = input("Input Product Name in exact characters -----> ")
        tr = tr.lower()
        coll_data.append(tr)
    recc_1 = adv_recc(coll_data[0],n)
    recc_2 = adv_recc(coll_data[1],n)
    recc_3 = adv_recc(coll_data[2],n)
    recc_4 = adv_recc(coll_data[3],n)
    recc_5 = adv_recc(coll_data[4],n)
    #print(recc_1[0])
    union = list(set(recc_1) | set(recc_2) | set(recc_3) | set(recc_4) | set(recc_5))
    print(union)
    print("CALCULATING SIMILARITIES IN SHOPPING PATTERN")
    #time.sleep(6)
    my_dict = {}
    for x in union:
        my_dict[x] = 1

    for x in union:
        if x in set(recc_1):
            my_dict[x] += 1
        if x in set(recc_2):
            my_dict[x] += 1
        if x in set(recc_3):
            my_dict[x] += 1
        if x in set(recc_4):
            my_dict[x] += 1
        if x in set(recc_5):
            my_dict[x] += 1

    my_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}
    print(my_dict)
    print(type(my_dict))
    top_recc = list(my_dict.keys())
    print("\n\n\n\n\n\n\n")
    print("******************RECOMMENDATIONS**********************")
    for x in range(n):
        print(top_recc[x])
    print("*******************************************************")
    print("\n\n\n\n\n\n\n")
    listed = list(my_dict.items())
    listed1 = listed[0:5]
    listed2 = listed[5:10]
    x,y = zip(*listed1)
    x1,y1 = zip(*listed2)
    x2,y2 = zip(*listed)
    plt.plot(x,y,'r-')
    plt.title('Relative Demand Plot')
    plt.plot(x1,y1,'y-')
    plt.plot(x2,y2,'go')
    plt.show()
    
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
print("************************************************MENU*******************************************************")
c = input("Enter 1 for Basic Product Recommendation \nEnter 2 for Advanced Product Recommendation\nEnter 3 for Collaborative Filtering(beta)\nEnter 4 to view Figures\nEnter your choice here -----> ")
if(c=='1'):
    tr = input("Input Product Name in exact characters -----> ")
    tr = tr.lower()
    n = input("Input Number of Recommendations Required -----> ")
    n = int(n)
    basic_recc(tr,n)
elif(c=='2'):
    tr = input("Input Product Name in exact characters -----> ")
    tr = tr.lower()
    n = input("Input Number of Recommendations Required -----> ")
    n = int(n)
    adv_recc(tr,n)
elif(c=='3'):
    n = input("Input Number of Recommendations Required -----> ")
    n = int(n)
    collaborative(n)
elif(c=='4'):
    show_figure()
else:
    print("Invalid Choice")
print("************************************************************************************************************")

#@author Arpan Sarkar
#START PGP MD5
#c5dbab9480173ef71df02e9b7721aa65277588f102243d14b1ed1ed0c38a09ef
#END PGP MD5
