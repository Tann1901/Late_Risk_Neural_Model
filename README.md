# Late_Risk_Neural_Model
An analysis of a smart supply chain dataset using machine learning and neural networks to predict late delivery risk and optimize operations.
## TABLE OF CONTENT
1. INTRODUCTION	
2. LITERATURE REVIEW	
3. DATA STRUCTURE	
4. METHODOLOGY
5. ANALYSIS & CONCLUSION
6. REFERENCE	

## 1. INTRODUCTION

In the fast-paced world of e-commerce, timely delivery is not just a competitive advantage; it is a fundamental expectation of customers. The ability to promise and fulfill delivery commitments is a critical component of customer satisfaction and retention. However, the e-commerce industry faces a myriad of challenges when it comes to meeting these delivery expectations, and late deliveries pose a significant risk to both customer loyalty and the overall success of DataCo Global.

This report delves into the intricate landscape of late delivery risk within DataCo Global. It seeks to comprehensively assess the factors contributing to late deliveries to mitigate these risks effectively. By understanding the root causes and consequences of late deliveries, we aim to empower the company to optimize its supply chain, and enhance customer experiences.

In the following sections, we will examine the scope and significance of late delivery risk, provide an overview of the methodology used for this analysis, and outline the structure of the report. 

## 2. LITERATURE REVIEW

In the rapidly evolving world of e-commerce, the precise prediction and effective management of delivery lateness risk are fundamental to ensuring customer satisfaction and optimizing operational efficiency.
Baryannis, Dani, and Antoniou (2019) conducted research that underscores the critical need not only to predict delivery lateness but also to ensure the interpretability of these predictions. Their paper introduces a supply chain risk prediction framework that has been applied to real-world multi-tier manufacturing supply chains. Notably, the research highlights a crucial trade-off between prediction performance and interpretability, a consideration that holds immense significance for making informed decisions within the realm of supply chain management.

Expanding on this foundation, Steinberg et al. (2023) contributed to the field by introducing a novel regression-based machine learning model tailored to predict the severity of delivery lateness. Their work goes a step further by demonstrating the feasibility of early prediction within the procurement process, thereby challenging the conventional wisdom of reducing dimensionality in high-dimensional input features. In the context of e-commerce, where comprehending not only the likelihood but also the potential impact of delivery delays is of paramount importance, this research takes a pioneering approach. Moreover, its focus on predicting delays early in the procurement process empowers e-commerce companies to proactively strategize and minimize disruptions.

The insights gleaned from both of these seminal papers provide invaluable guidance for the implementation of neural networks in predicting delivery lateness risk within the e-commerce landscape. Baryannis et al.'s emphasis on interpretability paves the way for the development of transparent neural network models that offer insights into the reasoning behind predictions. In parallel, Steinberg et al.'s concentration on regression models aligns with the concept of harnessing neural networks to estimate the severity of delivery lateness, transcending binary predictions. By amalgamating these insights into the design and application of neural network models, the e-commerce sector stands poised to enhance its ability to forecast and manage delivery risk, thereby delivering a seamless and dependable shopping experience for consumers.

## 3. DATA STRUCTURE
The data was obtained from Kaggle consisting of a total of 52 columns and 180,519 rows of data. it had a combination of string, integer,  and float data-types. It contained information about the organization’s customers including their first and last name, unique ID, country of origin, e-mail, and customer segment (Consumer, Corporate, Home Office). Although important for delivery purposes, they were non-essential for model-building purposes and subsequently dropped as feature/independent variables.
The data also contained information about orders including originating and destination cities, a unique order identifier, the order’s price, and quantity as well as its status. Also on record was the date the order was placed and the value of the discount offered on the order.  These attributes were valuable and taken into account during the exploratory analysis as well as deemed relevant for constructing the model.
Furthermore, the data contained product-related information including a unique product identifier, category, description, image, name, price, and status with the latter indicating whether these products were in stock (1) or out of stock (0).
Lastly, we had shipping-related data which included details such as shipping date, mode (Standard Class, First Class, Second Class, Same Day), delivery status (Advance shipping, Late delivery, Shipping canceled, Shipping on time) , actual number of days for shipping and predicted shipping duration. The model was built to predict late delivery risk (dependent, output variable) encoded as 1= late delivery, 0= not late. The model aims to forecast the likelihood of delayed order deliveries by leveraging patterns acquired from the selected feature variables.
| Data Original Input                                    |
|-------------------------------------------------------|
| 'Type', 'Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer', 'Delivery Status', 'Late_delivery_risk', 'Category Id', 'Category Name', 'Customer City', 'Customer Country', 'Customer Email', 'Customer Fname', 'Customer Id', 'Customer Lname', 'Customer Password', 'Customer Segment', 'Customer State', 'Customer Street', 'Customer Zipcode', 'Department Id', 'Department Name', 'Latitude', 'Longitude', 'Market', 'Order City', 'Order Country', 'Order Customer Id', 'order date (DateOrders)', 'Order Id', 'Order Item Cardprod Id', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id', 'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Order Item Total', 'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status', 'Order Zipcode', 'Product Card Id', 'Product Category Id', 'Product Description', 'Product Image', 'Product Name', 'Product Price', 'Product Status', 'shipping date (DateOrders)', 'Shipping Mode' |

| Variables Used                                         |
|-------------------------------------------------------|
| 'Type', 'Benefit per order', 'Sales per customer', 'Late_delivery_risk', 'Category Id', 'Category Name', 'Customer City', 'Customer Country', 'Customer Id', 'Customer Segment', 'Customer State', 'Customer Zipcode', 'Department Id', 'Department Name', 'Market', 'Order City', 'Order Country', 'Order Customer Id', 'Order Id', 'Order Item Cardprod Id', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id', 'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Order Item Total', 'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status', 'Product Card Id', 'Product Category Id', 'Product Name', 'Product Price', 'Shipping Mode', 'order_yr', 'order_month', 'order_day', 'order_hour', 'late_days' |

| Variables Selected for Neural Network Model           |
|-------------------------------------------------------|
| 'Order Status', 'Shipping Mode', 'late_days'          |

| Outcome                                               |
|-------------------------------------------------------|
| 'Late_delivery_risk'                                   |

Table 3.1. Data Table
## 4. METHODOLOGY
4.1. Data Preparation
In order to analyze valid data easily, we need to clean the dataset to ensure its relevance and completeness.

First of all, some columns that are not relevant to the analysis, such as "Customer Email", "Customer Password", "Latitude", "Longitude", "Product Image", etc., should be deleted from the dataset. Secondly, by analyzing the missing values, we find that there are a large number of missing data in “Order Zip Code” and “Product Description”, which we choose to delete, and there are very few missing values in “Customer Zipcode”, which we fill with 0. Next, we found that "order date (DateOrders)" is expressed in terms of year, month, day and time, which is inconvenient for our subsequent analysis, so we converted it into four separate columns: year, month, day, hour. Finally, we found that 'Days for shipping (real)' and 'Days for shipment (scheduled)' can help us to figure out how many days the product's shipment is late, which is more intuitive and easy to analyze, so we created a new column “late_days” to store the information we get.

The data cleaning and processing steps resulted in a complete and useful dataset that provided the basis for subsequent modeling and analysis.

#### Data view
```
df = pd.read_csv('DataCo.csv', encoding='ISO-8859-1')
df
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/4daf7531-aab5-4414-a173-410b7f9dcc81)

```
df.columns
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/b5646bb1-63c3-4e44-b7b4-6eb378a11ff9)

#### Data Cleaning

Check for missing values
```
missing_values = df.isnull().sum()
print(missing_values)
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/2bc402d9-cdc3-468c-b8ad-6ddb4f6b5903)


Dropping unnecessary columns
```
df = df.drop(['Customer Email'
              , 'Customer Fname'
              , 'Customer Lname'
              ,'Product Status'
              ,'Customer Password'
              ,'Customer Street'
              ,'Customer Fname'
              ,'Customer Lname'
              ,'Latitude'
              ,'Longitude'
              ,'Product Description'
              ,'Product Image'
              ,'Order Zipcode'
              ,'shipping date (DateOrders)'], axis=1)
```

Splitting Order dates and creating new columns
```
df['order_yr']= pd.DatetimeIndex(df['order date (DateOrders)']).year
df['order_month'] = pd.DatetimeIndex(df['order date (DateOrders)']).month
df['order_day'] = pd.DatetimeIndex(df['order date (DateOrders)']).weekday
df['order_hour'] = pd.DatetimeIndex(df['order date (DateOrders)']).hour
```
Drop 'order date (DateOrders)' after splitted
```
df = df.drop(['order date (DateOrders)'], axis=1)
```
Fill null in Customer Zipcode
```
df['Customer Zipcode'] = df['Customer Zipcode'].fillna(0)
```
Days of Late
```
df['late_days'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
```

#### Data Visualization 
Check proportion of Late Delivery Risk
```
df['Late_delivery_risk'].value_counts().plot.pie(legend = ["0", "1"])
```
![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/849e2ef5-05a6-4b5c-bb5d-6aeabff05614)

##### Check status of orders
```
status = df.groupby('Type')['Order Status'].value_counts()
status
```
![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/b0f6e36a-4ee5-4b25-9ed4-c0f5ff8460ee)

##### Check Order Country and Market
```
top_15_countries = df['Order Country'].value_counts().nlargest(15)

plt.figure(figsize=(15, 7))
top_15_countries.plot(kind='bar', title="Order Origin")
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/ee84fc46-667d-4a46-8adf-890794b487c1)

```
plt.figure(2)
df['Market'].value_counts().nlargest(10).plot.bar(figsize=(15,7), title="Market")
plt.xticks(rotation = 45)
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/b4464d26-f675-43db-91ea-42300aab762c)

##### Check shipping mode
```
shipping_mode = df['Shipping Mode'].value_counts()

plt.figure(figsize=(15, 7))
shipping_mode.plot(kind='bar', title="Shipping mode")
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/ff37c63b-5c22-49fb-8431-159b10ac15df)

##### Check Department of Goods
```
department_counts = df['Department Name'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Department Distribution")

plt.show()
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/f3f4ce60-4293-4fdd-b097-ae9f5dad9edc)

##### Check if loss after promo
```
loss = len(df[df['Benefit per order'] <= 0])
earn = len(df[df['Benefit per order'] > 0])

categories = ['Loss', 'Earn']
counts = [loss, earn]

plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['red', 'green'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Loss vs. Earn')

plt.show()
```


![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/473279b5-340a-481a-bfc9-858c4de48c03)

##### Correlation Matrix
```
correlation_matrix = df.corr()
plt.figure(figsize=(30, 24))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/7d45d19a-7da0-4a51-9764-c859d4fed761)


### 4.2 Data Encoding and Splitting
To facilitate the integration of categorical variables into neural network models, a systematic encoding process was undertaken. By looking at the datatype of the data, we find categorical variables and convert them to numerical variables using the labeled coding method, which assigns a unique numerical label to each of the categorical variables while maintaining the sequential relationship between the categories. This numerical representation allows the neural network to process and learn the categorical information and helps to improve the overall effectiveness of the model in capturing the complex relationships in the data.

Data splitting is a pivotal step in our machine learning workflow, allowing us to assess our model's performance effectively. We begin by creating a Random Forest classifier for feature selection, utilizing the chosen features as predictors for our model. To ensure unbiased evaluation, the data was divided into training data (60%) and validation data (40%). The StandardScaler() function was used to scale the Variables and create a valid model. 

##### Variable Selection and Encode 
```
train_df = df.copy()
train_df. columns
train_df.dtypes
```
##### Encode all Categorical Object Variables
```
label_encoder = LabelEncoder()
columns_to_encode = ['Type'
                       , 'Category Name'
                       , 'Customer City'
                       , 'Customer Country'
                       , 'Customer Segment'
                       , 'Customer State'
                       ,'Delivery Status'
                       , 'Department Name'
                       , 'Market'
                       , 'Order City'
                       , 'Order Country'
                       , 'Order Region'
                       , 'Order State'
                       , 'Order Status'
                       , 'Product Name'
                       , 'Shipping Mode']

for column in columns_to_encode:
    train_df[column] = label_encoder.fit_transform(train_df[column])
```
##### Redo the Correlation Matrix to see all veriables after encoded
```
correlation_matrix = train_df.corr()
plt.figure(figsize=(30, 24))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/03c74331-53af-4014-b6dc-553b3752f283)

##### Choosing predictors and outcome
Drop "Days for shipping (real)", "Days for shipment (scheduled)" as already created late_days column
```
train_df.drop(["Days for shipping (real)", "Days for shipment (scheduled)", "Delivery Status"], axis=1, inplace=True)
```
##### Using Random Forest to figure which variable to select for X - predictors
```
# Create a Random Forest classifier for feature selection
rf = RandomForestClassifier(n_estimators=45, random_state=42)

# Fit the classifier to your data for feature selection
X = train_df.drop(columns=['Late_delivery_risk'])
y = train_df['Late_delivery_risk']
rf.fit(X, y)

# Use feature importances to select the top N features
num_features_to_select = 5
selector = SelectFromModel(rf, max_features=num_features_to_select)
selector.fit(X, y)

# Get the selected feature indices and feature names
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

# Select only the top features
X_new = X[selected_feature_names]
```

```
# Define predictors (selected features) and outcome
predictors = X_new
outcome = y

# Split the data for training and validation
train_X, valid_X, train_y, valid_y = train_test_split(predictors, outcome, test_size=0.4, random_state=0)
```

### 4.3 Model Development
The objective is to develop a robust predictive model that can effectively classify deliveries as either 'late' or 'on time.'

The choice of a Random Forest classifier as the model for feature selection is motivated by its versatility in handling both categorical and numerical data, as well as its feature importance estimation.

The machine learning model, in this case, a neural network classifier, is trained using the selected features from the training set：

The activation function used in this layer is the logistic (or sigmoid) function, which is a common choice for binary classification problems. This function will open the gate for the model to predict whether the input data belongs to class 0 or class 1. The solver 'lbfgs' is used for optimization, and the random state is set for reproducibility. In this case, we have opted for a modest number of 3 hidden nodes in two layers for several reasons:

- After trying between 1 layer, 2 layers and 3 layers, the choice of 2 layers with 3 nodes each received the highest accuracy scores as of 0.8629 (training set) and 0.8616(valid set) in compared to 0.8325 (training set) and 0.8325 ( valid set) of 1 layer and               0.7203 (training set) and 0.7177 (valid set) of 3 layers.

| Performance                         | Precision | Setting                            |
|------------------------------------|-----------|------------------------------------|
| Training Set (One layer - 3 hidden nodes/layer) | 0.8325    | Valid Set                        |
| Training Set (Two layers - 3 hidden nodes/layer) | 0.8629   | Valid Set                        |
| Training Set (Three layers - 3 hidden nodes/layer) | 0.7203  | Valid Set                        |

Table 4.3. Comparison of hidden nodes and layers selection

- We continue to opt for 5 fold Cross Validation and received the result of Average Accuracy: 97.86% and Standard Deviation: 0.05. This proves that, on average, our model is correct in its predictions about 97.86% of the time. Also, the standard deviation                 measures the variability or spread of accuracy across the five folds, 0.05 is relatively low, suggesting that our model's performance is quite stable across different data splits.

- Our dataset and problem are relatively simple, and overly complex models may lead to overfitting. A small number of nodes helps maintain model simplicity. A smaller architecture reduces computational demands during training, making it more efficient, especially        for small to medium-sized datasets.

- The choice of 3 hidden nodes serves as an initial experiment. As we further explore Deep Learning, we would explore more complex architectures with additional hidden layers and nodes.

#### One layer - 3 nodes
```
# train neural network with 3 hidden nodes *
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

clf = MLPClassifier(hidden_layer_sizes=(3), activation='logistic', solver='lbfgs', random_state=1)
clf.fit(train_X_scaled, train_y.values)

# training performance
classificationSummary(train_y, clf.predict(train_X))
# validation performance
classificationSummary(valid_y, clf.predict(valid_X))
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/9e26d353-3dc7-4955-84a6-60f339ced684)


#### Two layers - 3 nodes
```
# train neural network with 2 hidden layers and a total of 6 nodes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Define the architecture with two hidden layers
clf = MLPClassifier(hidden_layer_sizes=(3, 3), activation='logistic', solver='lbfgs', random_state=1)
clf.fit(train_X_scaled, train_y.values)

# training performance
classificationSummary(train_y, clf.predict(train_X))
# validation performance
classificationSummary(valid_y, clf.predict(valid_X))
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/fbff71a1-ab79-4bcb-9369-1f5028686ebc)

#### 5 folded Cross Validation 

```
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Define your classifier and preprocessing steps (e.g., scaling) here
clf = MLPClassifier(hidden_layer_sizes=(3, 3), activation='logistic', solver='lbfgs', random_state=1)
scaler = StandardScaler()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, train_X, train_y, cv=5)  # 5-fold cross-validation

# Print the average accuracy and standard deviation
print("Average Accuracy: {:.2f}%".format(100 * cv_scores.mean()))
print("Standard Deviation: {:.2f}".format(100 * cv_scores.std()))
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/4d2371b4-9cf6-4f74-84cd-a7d94d2cf246)

#### Checking recall, f1 and ROC_AUC
##### Train Set
```
# Create an instance of MLPClassifier and a StandardScaler
clf = MLPClassifier(hidden_layer_sizes=(3, 3), activation='logistic', solver='lbfgs', random_state=1)
scaler = StandardScaler()

# Fit the model with the training data
train_X_scaled = scaler.fit_transform(train_X)
clf.fit(train_X_scaled, train_y)

# Make predictions on the training data
train_predictions = clf.predict(train_X)

# Calculate Recall on training data
train_recall = recall_score(train_y, train_predictions)

# Calculate F1-Score on training data
train_f1 = f1_score(train_y, train_predictions)

# Calculate ROC AUC on training data
train_roc_auc = roc_auc_score(train_y, clf.predict_proba(train_X)[:, 1])

# Display the results for the training data
print("Train Recall:", train_recall)
print("Train F1-Score:", train_f1)
print("Train ROC AUC:", train_roc_auc)

# Generate the ROC curve on training data (optional)
fpr_train, tpr_train, thresholds_train = roc_curve(train_y, clf.predict_proba(train_X)[:, 1])
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/89cb3d5c-9671-429c-a12c-5ababa2872a6)

##### Valid Set
```
# Create an instance of MLPClassifier and a StandardScaler
clf = MLPClassifier(hidden_layer_sizes=(3, 3), activation='logistic', solver='lbfgs', random_state=1)
scaler = StandardScaler()

# Fit the model with the training data
train_X_scaled = scaler.fit_transform(train_X)
clf.fit(train_X_scaled, train_y)

# Make predictions on the validation data
predictions = clf.predict(valid_X)

# Calculate Recall on validation data
recall = recall_score(valid_y, predictions)

# Calculate F1-Score on validation data
f1 = f1_score(valid_y, predictions)

# Calculate ROC AUC on validation data
roc_auc = roc_auc_score(valid_y, clf.predict_proba(valid_X)[:, 1])

# Display the results
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC AUC:", roc_auc)

# Generate the ROC curve on validation data (optional)
fpr, tpr, thresholds = roc_curve(valid_y, clf.predict_proba(valid_X)[:, 1])
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/b2f2521d-f00f-470d-bd8c-2f5e8d2199bb)
#### Three layers - 3 nodes
```
# train neural network with 2 hidden layers and a total of 6 nodes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Define the architecture with two hidden layers
clf = MLPClassifier(hidden_layer_sizes=(3, 3,3), activation='logistic', solver='lbfgs', random_state=1)
clf.fit(train_X_scaled, train_y.values)

# training performance
classificationSummary(train_y, clf.predict(train_X))
# validation performance
classificationSummary(valid_y, clf.predict(valid_X))
```

![image](https://github.com/Tann1901/Late_Risk_Neural_Model/assets/108020327/2d4afd1e-9002-4437-824b-b0f36c919460)

## 5. ANALYSIS & CONCLUSION
### 5.1. Conclusion
Break down the confusion matrix to see where the model is making false positives and false negatives. This can help identify areas for improvement.
| Performance                         | Precision | Recall | F1       | ROC      | 
|------------------------------------|-----------|--------|----------|----------| 
| Training Set (One layer - 3 hidden nodes/layer) | 0.8629    | 0.7875 | 0.8632 | 0.9266   | 
| Valid Set (One layer - 3 hidden nodes/layer)    | 0.8616    | 0.7856 | 0.8612 | 0.9250   | 

Table 5.1. Confusion Matrix, Recall, F1-Score, ROC AUC Results

In the evaluation of our neural network model's performance, we achieved an accuracy of 86.16% in distinguishing between two classes (0 and 1). Notably, the model demonstrated improved precision and recall, reducing false positives and false negatives. The recall score reflects the model's ability to capture true positives, while the F1-Score strikes a balance between precision and recall. Furthermore, the ROC AUC score of 0.93 indicates the model's effectiveness in classifying positive instances. These results highlight the promising potential of our model for the given task.

### 5.2 Business Impact Analysis
The implementation of our neural network model offers significant business benefits by mitigating late delivery risk. By accurately predicting and preventing late deliveries, businesses can enhance customer satisfaction, reduce costly penalties, and strengthen their brand reputation. This predictive tool empowers companies to allocate resources more efficiently, streamline supply chain operations, and ultimately drive increased profitability. Furthermore, proactive measures to reduce late deliveries can lead to higher customer retention rates and a competitive edge in the market.

### 5.3 Future Directions and Recommendations
In considering future directions, it's essential to continuously refine and expand our neural network model. This includes exploring more advanced architectures, leveraging larger datasets, and investigating emerging techniques in deep learning. Additionally, focusing on real-time or near-real-time monitoring of delivery statuses could enhance the model's predictive capabilities. Furthermore, collaboration with logistics and supply chain experts can provide valuable insights for further improvements in delivery risk management. Continuous model evaluation and adaptation to evolving business needs remain critical, ensuring sustained success and resilience in addressing late delivery risk. (Taye, M. M.,2023)


## 6. REFERENCE

Baryannis, G., Dani, S., & Antoniou, G. (2019). Predicting supply chain risks using machine learning: The trade-off between performance and interpretability. Future Generation Computer Systems, 101(December 2019), 993-1004. https://www.sciencedirect.com/science/article/pii/S0167739X19308003

Steinberg, F., Burggräf, P., Wagner, J., Heinbach, B., Saßmannshausen, T., & Brintrup, A. (2023). A novel machine learning model for predicting late supplier deliveries of low-volume-high-variety products with application in a German machinery industry. Supply Chain Analytics, Volume 1(March 2023), N/A. https://www.sciencedirect.com/science/article/pii/S294986352300002X

Taye, M. M. (2023). Understanding of Machine Learning with Deep Learning: Architectures, Workflow, Applications and Future Directions. Computers, 12(5), 91. https://www.mdpi.com/2073-431X/12/5/91

Tiwari, S. (2019). DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS. Kaggle. https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/data
