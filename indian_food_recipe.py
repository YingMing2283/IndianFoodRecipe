#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors 
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import streamlit

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


df=pd.read_csv("IndianFoodDatasetCSV.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# ## Removing null values & unwanted columns

# In[6]:


df = df[df['Ingredients'].notna()]


# In[7]:


df.isna().sum()


# In[8]:


df.head()


# In[9]:


df = df.drop(['Srno','RecipeName','Ingredients','Instructions','TranslatedInstructions','URL'],axis=1)


# In[10]:


df.head()


# In[11]:


df['TranslatedRecipeName'].nunique()


# In[12]:


df.TranslatedRecipeName.value_counts()


# In[13]:


df.Cuisine.value_counts()


# In[14]:


df2 = df[df['Cuisine'].isin(['Indian','North Indian Recipes','South Indian Recipes','Bengali Recipes',
                            'Maharashtrian Recipes','Kerala Recipes','Tamil Nadu',
                            'Karnataka','Fusion','Rajasthani','Andhra','Gujarati Recipes',
                            'Goan Recipes','Punjabi','Chettinad','Kashmiri','Mangalorean',
                            'Indo Chinese','Parsi Recipes','Awadhi','Oriya Recipes','Sindhi',
                            'Konkan','Mughlai','Bihari','Assamese','Hyderabadi','North East India Recipes',
                            'Himachal','Sri Lankan','Udupi','Coorg','Uttar Pradesh','North Karnataka',
                            'Coastal Karnataka','Malabar','Lucknowi','South Karnataka','Malvani',
                            'Nagaland','Uttarakhand-North Kumaon','Kongunadu','Haryana','Jharkhand'
                            ])]


# In[15]:


df2.Cuisine.value_counts()


# In[16]:


df2.head()


# In[17]:


df2.Course.value_counts()


# In[18]:


df2.Diet.value_counts()


# In[46]:


# For regular expressions
import re
# For handling string
import string
# For performing mathematical operations
import math
import nltk

# ## Cosine Similarity & Euclidean Distance and Linear Kernel

# In[47]:

vocabulary = nltk.FreqDist()
# This was done once I had already preprocessed the ingredients
for ingredients in df2['TranslatedIngredients']:
    ingredients = ingredients.split()
    vocabulary.update(ingredients)
for word, frequency in vocabulary.most_common(200):
    print(f'{word};{frequency}')


# In[48]:


for index,text in enumerate(df2['TranslatedIngredients'][35:40]):
  print('Ingredient %d:\n'%(index+1),text)


# In[49]:


#lowercase
df2['cleaned_ingredient']=df2['TranslatedIngredients'].apply(lambda x: x.lower())

print(df2['cleaned_ingredient'])


# In[50]:


#Remove digits and words containing digits
df2['cleaned_ingredient']=df2['cleaned_ingredient'].apply(lambda x: re.sub('\w*\d\w*','', x))

print(df2['cleaned_ingredient'])


# In[51]:


# remove - / 
df2['cleaned_ingredient'] = df2['cleaned_ingredient'].apply(lambda x: re.sub(r'[-/+]', ' ', x))

print(df2['cleaned_ingredient'])


# In[52]:


# remove punctuation
df2['cleaned']=df2['cleaned_ingredient'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

print(df2['cleaned'])


# In[53]:


def remove_units(text):
    units_to_remove = ["pinch","inch","tbsp","cup", "cups", "tablespoon","tablespoons","teaspoons", "teaspoon","ounces", "ounce","gram","grams", "kg", "मिटर"]  # Add other units as needed
    for unit in units_to_remove:
        text = text.replace(unit, "")
    return text

df2['cleaned2'] = df2['cleaned'].apply(remove_units)

# Display the DataFrame
print(df2['cleaned2'])


# In[54]:


# Removing extra spaces
df2['cleaned2']=df2['cleaned2'].apply(lambda x: re.sub(' +',' ',x))

print(df2['cleaned2'])


# In[55]:


for index,text in enumerate(df2['cleaned2'][35:40]):
  print('Ingredient %d:\n'%(index+1),text)


# In[56]:


# tokenization
df2['cleaned_tokens'] = df2['cleaned2'].apply(lambda x: x.split())
print (df2['cleaned_tokens'])


# In[57]:


#Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
df2['cleaned_lemmas'] = df2['cleaned_tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

print (df2['cleaned_lemmas'])


# In[58]:


# remove stopwords
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df2['cleaned_no_stopwords'] = df2['cleaned_lemmas'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

print (df2['cleaned_no_stopwords'])


# In[59]:


df2 = df2.dropna(subset=['cleaned_no_stopwords'])


# In[60]:


# words frequency analysis
from collections import Counter

word_frequencies = Counter([word for words in df2['cleaned_no_stopwords'] for word in words])

print (word_frequencies)


# In[61]:


# Assuming df2 is your DataFrame
grouped_by_diet = df2.groupby('Diet')

# Example: Get word frequencies for each diet group
word_frequencies_by_diet = grouped_by_diet['cleaned_no_stopwords'].apply(lambda x: Counter([word for word_list in x for word in word_list]))
print(word_frequencies_by_diet)


# In[62]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(word_frequencies)

print (tfidf_matrix)


# In[63]:


import time

# Record start time
start = time.time()

#Explore Similarity Metrics:
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

print (similarity_matrix)

print("Time taken: %s seconds" % (time.time() - start))


# In[64]:


# Record start time
start = time.time()

euclidean_distance_matrix = euclidean_distances(tfidf_matrix)

print (euclidean_distance_matrix)

print("Time taken: %s seconds" % (time.time() - start))


# In[65]:


# Record start time
start = time.time()

linear_kernel_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

print (linear_kernel_matrix)

print("Time taken: %s seconds" % (time.time() - start))


# In[66]:


if 'cleaned_no_stopwords' not in df2.columns or df2['cleaned_no_stopwords'].isnull().all():
    print("Error: 'cleaned_no_stopwords' column not found or is empty.")
else:

    ingredient = "karela"
    ingredient_indices = df2[df2['cleaned_no_stopwords'].astype(str).str.contains(ingredient, case=False)].index

    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
    else:
        # Display a few records to confirm the data
        print(df2.loc[ingredient_indices, 'cleaned_no_stopwords'])


# In[67]:


print(df2.columns)


# In[68]:


df2.head()


# In[71]:


df.head()


# In[72]:


df2.head()


# In[79]:


df3=pd.read_csv("C:/Users/user/Desktop/IndianFoodDatasetCSV.csv")

df3.head()


# In[85]:


df3 = df3.drop(['Srno', 'RecipeName', 'Ingredients', 'TranslatedIngredients', 'PrepTimeInMins',
                'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine', 'Course', 'Diet', 'Instructions'],
               axis=1)
df3 = pd.DataFrame(df3)
df3.head()


# In[87]:


df2 = pd.merge(df2, df3, on='TranslatedRecipeName', how='left')
df2.head()


# In[94]:


df2 = df2.rename(columns={'TranslatedRecipeName':'Recipe Name', 'TranslatedIngredients':'Ingredients',
                          'TranslatedInstructions':'Instructions', 'CookTimeInMins':'Cooking Time (Min)',
                         'PrepTimeInMins':'Preparation Time (Min)', 'TotalTimeInMins':'Total (Min)',})


df2.head()


# In[96]:


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_similarity_matrix(word_frequencies):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(word_frequencies)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

def recommend_food(ingredient, diet,course, similarity_matrix, df2):
    nan_indices = df2[df2['cleaned_no_stopwords'].isna() | df2['Diet'].isna() | df2['Course'].isna()].index
    df2 = df2.drop(index=nan_indices)

    ingredient_indices = df2[df2['cleaned_no_stopwords'].apply(lambda x: ingredient in x)].index

    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
        return None

    # Get the similarity scores for the corresponding row
    idx = ingredient_indices[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Calculate weighted similarity scores based on the presence of the ingredient
    weighted_scores = [
        (i, score + 0.5 if ingredient in df2['cleaned_no_stopwords'].iloc[i] else score + 0.3 if diet in df2['Diet'].iloc[i] else score+ 0.1 if course in df2['Course'].iloc[i] else score)
        for i, score in similarity_scores]

    # Sort the food items by weighted similarity score
    sorted_items = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

    # Extract the indices of recommended food items (excluding the input ingredient)
    recommended_indices = [i[0] for i in sorted_items if i[0] != idx]

    # Get the entire rows for recommended items from the original DataFrame
    recommended_data = df2.iloc[recommended_indices]
        
    return recommended_data

def main():
    st.title("Indian Food Recommender System")

    # Sidebar with user input
    ingredient_input = st.sidebar.text_input("Enter an ingredient:")

    # Dropdown for selecting diet
    diet_options = df2['Diet'].unique()
    diet_options = sorted(diet_options)
    diet_input = st.sidebar.selectbox("Select a diet:", diet_options)

    # Dropdown for selecting course
    course_options = df2['Course'].unique()
    course_options = sorted(course_options)
    course_input = st.sidebar.selectbox("Select a course:", course_options)


    if st.sidebar.button("Recommend"):
        if ingredient_input or diet_input or course_input:
            
            grouped_by_diet = df2.groupby('Diet')
            word_frequencies_by_diet = grouped_by_diet['cleaned_no_stopwords'].apply(lambda x: Counter([word for word_list in x for word in word_list]))

            # Use your recommendation function
            similarity_matrix = load_similarity_matrix(word_frequencies)
            recommended_data = recommend_food(ingredient_input, diet_input, course_input, similarity_matrix, df2)

            if recommended_data is not None:
                st.write("Top 3 Recommendations:")
                st.table(recommended_data[['Recipe Name', 'Ingredients', 'Preparation Time (Min)', 'Cooking Time (Min)', 'Total (Min)', 'Servings', 'Cuisine', 'Course', 'Diet','Instructions', 'URL']].head(3))
            else:
                st.warning("No recommendations found for the given inputs.")
        else:
            st.warning("Please enter values for ingredient, diet, and course.")

if __name__ == "__main__":
    main()


# In[ ]:




