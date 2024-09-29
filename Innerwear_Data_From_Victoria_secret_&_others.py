#!/usr/bin/env python
# coding: utf-8

# # About Dataset
# For this project, Data set is taken from Kaggle . They were created by extracting data from from the popular retail sites via PromptCloud's data extraction solutions:
# Sites Covered
# - Amazon
# - Victoria's Secret
# - Btemptd
# - Calvin Klein
# - Hanky Panky
# - American Eagle
# - Macy's
# - Nordstrom
# - Topshop USA

# # Goal

# - How does the pricing differ depending on the brand?
# - Popular brand based on the review.
# - Most popular product of brand based on rating 
# - What are the most common color used by different brands?

# In[1]:


import pandas as pd
df=[]
amazon_data=pd.read_csv("amazon_com.csv")
df.append(amazon_data)

victoria_data = pd.read_csv("victoriassecret_com.csv")
df.append(victoria_data)

btemptd_data =pd.read_csv("btemptd_com.csv")
df.append(btemptd_data)

calvin_Klein_data=pd.read_csv("calvinklein_com.csv")
df.append(calvin_Klein_data)

Hanky_Panky_data=pd.read_csv("hankypanky_com.csv")
df.append(Hanky_Panky_data)

American_Eagle_data=pd.read_csv("ae_com.csv")
df.append(American_Eagle_data)

Macys_data=pd.read_csv("macys_com.csv")
df.append(Macys_data)

Nordstrom_data=pd.read_csv("shop_nordstrom_com.csv")
df.append(Nordstrom_data)

Topshop_USA_data=pd.read_csv("us_topshop_com.csv")
df.append(Topshop_USA_data)


# In[2]:


# combine all the data into single dataframe
final_data = pd.concat(df, ignore_index=True)
final_data


# In[3]:


final_data.info()


# In[4]:


final_data.columns


# In[5]:


# Check unique values for every column in the DataFrame
for column in final_data.columns:
    unique_values = final_data[column].unique()
    print(f"Unique values in '{column}': {unique_values}\n")


# # Exploratory Data Analysis

# #### 1.Checking for Null values

# In[6]:


final_data.isnull().any()


# Hence there are no null values present in the Dataset

# #### 2.Checking for duplicates

# In[7]:


final_data.duplicated()


# In[8]:


# Rows that are duplicated
duplicated_rows = final_data[final_data.duplicated()]
print(duplicated_rows)
print(final_data.duplicated().sum())


# #### 3.Dropping Duplicated values

# In[9]:


# dropping the duplicated rows
final_data.drop_duplicates(inplace = True)
# checking if they are removed
final_data.duplicated().sum()


# #### 4. Cells with str NULL

# - Here in this dataset there are no null values(empty cells) but there are cells with values as  NULL(style_attributes,review_count,rating,color )
# - Now we are going to check how many value NULL are there.

# In[10]:


def missing_value(data):
    missing_value_stats = (data.isnull().sum() / len(data)*100)
    missing_value_count = sum(missing_value_stats > 0)
    print("Number of columns with missing values:", missing_value_count)
    if missing_value_count != 0:
        print("\nMissing percentage:")
        print(missing_value_stats)
    else:
        print("No missing data")
missing_value(final_data)


# - Here style_attributes gives high percentage of missing values (79.76%), hence we are dropping this column.
# - Rating (44.752945) and review_count (47.554364) are important for analysis hence we are imputing them.

# In[11]:


# dropping style_attributes
final_data.drop(columns=['style_attributes'], inplace=True)
final_data


# In[12]:


print(final_data[['rating', 'review_count']].isnull().sum())


# #### 5. Outlier Detection 
# - Before imputing we are plotting boxplot for checking the  outliers .

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("Set1")
plt.figure(figsize=(10, 5))

# Box Plot for 'rating'
plt.subplot(1,2,1)
sns.boxplot(x=final_data['rating'].dropna())
plt.title('Box Plot for Ratings')

# Box Plot for 'review_count'
plt.subplot(1,2,2)
sns.boxplot(x=final_data['review_count'].dropna())
plt.title('Box Plot for Review Counts')

plt.tight_layout()
plt.show()


# #### 6.Skewness of Rating and review count

# In[14]:


print("Skewness of Ratings:", final_data['rating'].skew())
print("Skewness of Review Counts:", final_data['review_count'].skew())


# - The skewness of the rating distribution is negative (left-skewed).
# - The skewness of the review count distribution is positive (right-skewed).
# - Outliers are present in both columns.
# - Hence we need to remove these outliers before imputing so that we can choose mean or median imputation.

# #### 7. Outlier removal

# In[15]:


def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[~((data[column] < lower_bound) | (data[column] > upper_bound))]

# Removing outliers from both 'rating' and 'review_count'
filtered_data = remove_outliers(final_data, 'rating')
filtered_data = remove_outliers(filtered_data, 'review_count')
filtered_data


# In[16]:


plt.figure(figsize=(10,5))
# Box Plot for 'rating'
plt.subplot(1,2,1)
sns.boxplot(x=filtered_data['rating'])
plt.title('Box Plot for Ratings-Without Outliers')

# Box Plot for 'review_count'
plt.subplot(1, 2, 2)
sns.boxplot(x=filtered_data['review_count'])
plt.title('Box Plot for Review Counts -Without Outliers')

plt.tight_layout()
plt.show()


# In[17]:


print("Skewness of Ratings:", filtered_data['rating'].skew())
print("Skewness of Review Counts:", filtered_data['review_count'].skew())


# - In both cases, using the median for imputation is better as it provides a more reliable central tendency measure in the presence of skewed distributions.

# #### 8.Median Imputation

# In[18]:


# Perform median imputation correctly
filtered_data['rating'].fillna(filtered_data['rating'].median(), inplace=True)
filtered_data['review_count'].fillna(filtered_data['review_count'].median(), inplace=True)


# #### 9.Mode imputation for Color

# In[19]:


filtered_data['color'].fillna(filtered_data['color'].mode()[0], inplace=True)


# #### 10. Standardize mrp &  price values

# In[20]:


# removing space if it exists in the data
filtered_data['mrp'] = filtered_data['mrp'].str.strip()
filtered_data['price'] = filtered_data['price'].str.strip()

# remove anything that is not a number or a decimal point
filtered_data['mrp'] = filtered_data['mrp'].replace({'[^0-9.]': ''}, regex=True)
filtered_data['price'] = filtered_data['price'].replace({'[^0-9.]': ''}, regex=True)
# using pd.to_numeric
filtered_data['mrp'] = pd.to_numeric(filtered_data['mrp'], errors='coerce')
filtered_data['price'] = pd.to_numeric(filtered_data['price'], errors='coerce')


# #### 11. Standardize Brand name

# In[21]:


def std_brand_names(brand_name):
    brand_name = brand_name.lower().replace("-", " ")
    if "hanky" in brand_name:
        brand_name = "Hanky Panky"
    elif "calvin" in brand_name:
        brand_name = "Calvin klein"
    elif "wacoal" in brand_name or "tempt" in brand_name:
        brand_name = "B.tempt'd"
    elif "victorias" in brand_name or "victoria's" in brand_name or "s" in brand_name:
        brand_name = "Victoria's secret"
    elif "aeo" in brand_name or "aerie" in brand_name:
        brand_name = "Aerie" 
    
    brand_name = brand_name.strip()
    
    return brand_name

filtered_data['brand_name'] = filtered_data['brand_name'].apply(std_brand_names)
print(filtered_data['brand_name'].value_counts())


# #### 12.Feature Engineering
# - Discount percentage feature would be more impactful for the analysis using 'mrp' and 'price'
# 

# In[22]:


print(filtered_data['mrp'].isna().sum(),'for mrp')
print(filtered_data['price'].isna().sum(),'for price')


# #### Even after cleaning mrp and price null values are present(90 and 102 respectively). Hence we are dropping those rows .

# In[23]:


filtered_data = filtered_data.dropna(subset=['mrp', 'price'])
print(filtered_data['mrp'].isna().sum(),'for mrp')
print(filtered_data['price'].isna().sum(),'for price')


# ### Feature engineering : Discount_percentage

# In[24]:


filtered_data['discount_percentage'] = ((filtered_data['mrp'] - filtered_data['price']) / filtered_data['mrp']) * 100
filtered_data


# #### Since we have dropped style attribute column , we will be proceeding with the following goals:
# - Price Analysis by Brand
# - Topic Modeling on Product Descriptions
# - Color Analysis by Brand
# - Ratings Analysis

# ## Price Analysis by Brand

# In[25]:


import seaborn as sns
price_analysis = filtered_data.groupby('brand_name')['price'].agg(['mean', 'median', 'min', 'max', 'count'])
price_analysis = price_analysis.reset_index()
print(price_analysis)
# Bar chart using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='brand_name', y='mean', data=price_analysis.sort_values('mean'), palette='Set3')
plt.title('Mean Price by Brand')
plt.xlabel('Brand Name')
plt.ylabel('Mean Price')
plt.xticks(rotation=45)
plt.show()


# In[42]:


#Box plot for brand_name and price
plt.figure(figsize=(12, 6))
sns.boxplot(x='brand_name', y='price', data=filtered_data)
plt.title('Price Distribution by Brand')
plt.xlabel('Brand Name')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()


# - The bar chart for Price analysis by brand name (mean price) shows that B.tempt'd is more expensive than other brands, followed by Hanky Panky and Victoria's Secret.
# - However, the box plot results indicate that Hanky Panky and Victoria's Secret have relatively wide price ranges compared to the other brands, with B.tempt'd following closely.
# - Overall, the results indicate that B.tempt'd, Hanky Panky, and Victoria's Secret are the most expensive brands. This could be due to factors such as product variety, material quality, brand recognition, or target market.

# ## Review Analysis by Brand

# In[28]:


# Group the data by 'brand_name' and calculate the sum of 'review_count' for each brand
review_count_by_brand = filtered_data.groupby('brand_name')['review_count'].sum().reset_index()
print(review_count_by_brand)

plt.figure(figsize=(5,5))
plt.pie(review_count_by_brand['review_count'], labels=review_count_by_brand['brand_name'], autopct='%1.1f%%', 
        startangle=180)
plt.title('Proportion of Total Review Count by Brand')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# - Pie chart indicates that Victoria's Secret is the most popular brand in the market in terms of the review count ( almost 80% of the total) followed by B.temp'd.

# # Popular product of brand based on rating

# In[29]:


avg_rating_by_category = filtered_data.groupby(['brand_name', 'product_category'])['rating'].mean().reset_index()
most_popular_categories = avg_rating_by_category.loc[avg_rating_by_category.groupby('brand_name')['rating'].idxmax()]
print(most_popular_categories)

#Visualize the results
plt.figure(figsize=(12, 6))
sns.barplot(x='rating', y='product_category', hue='brand_name', data=most_popular_categories, palette='viridis')
plt.title('Most Popular Product Category by Brand Based on Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Product Category')
plt.legend(title='Brand Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()




# - Aerie's Cheekies is the most popular product category by brand based upon the average rating(4.8).
# - Victoria's secret's Lace-Trim Thong Panty is the second most popular followed by B.tempt'd Bridal Lingerie.
# - All of the brand have rating between 4.3 to 4.8

# # Common color used by the Brand

# In[38]:


color_counts = filtered_data.groupby(['brand_name', 'color']).size().reset_index(name='count')
most_common_colors = color_counts.loc[color_counts.groupby('brand_name')['count'].idxmax()]
print(most_common_colors)
# Visualize the results
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='color', hue='brand_name', data=most_common_colors, palette='viridis')
plt.title('Most Common Colors Used by Different Brands')
plt.xlabel('Count of Products')
plt.ylabel('Color')
plt.legend(title='Brand Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# - Black is a predominant color across multiple brands like  B.tempt'd ,Calvin klein ,Hanky Panky ,Victoria's secret and aerie.
# - Some unique colors are observed in smaller quantities for certain brands, such as "Star White" for Vanity fair.
#                  
#   

# # Correlation Analysis

# In[46]:


numerical_columns = ['mrp', 'price', 'rating', 'review_count', 'discount_percentage']
correlation_matrix = filtered_data[numerical_columns].corr()

# visualize
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

correlation_matrix


# - Strong positive correlation between MRP and Price.
# - Weak correlation between rating and other variables.
# - Moderate Correlation between Discount and price.

# In[ ]:




