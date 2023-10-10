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
Data Original Input
       'Type', 'Days for shipping (real)', 'Days for shipment (scheduled)',
       'Benefit per order', 'Sales per customer', 'Delivery Status',
       'Late_delivery_risk', 'Category Id', 'Category Name', 'Customer City',
       'Customer Country', 'Customer Email', 'Customer Fname', 'Customer Id',
       'Customer Lname', 'Customer Password', 'Customer Segment',
       'Customer State', 'Customer Street', 'Customer Zipcode',
       'Department Id', 'Department Name', 'Latitude', 'Longitude', 'Market',
       'Order City', 'Order Country', 'Order Customer Id',
       'order date (DateOrders)', 'Order Id', 'Order Item Cardprod Id',
       'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id',
       'Order Item Product Price', 'Order Item Profit Ratio',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status',
       'Order Zipcode', 'Product Card Id', 'Product Category Id',
       'Product Description', 'Product Image', 'Product Name', 'Product Price',
       'Product Status', 'shipping date (DateOrders)', 'Shipping Mode


Variables Used
       'Type', 'Benefit per order', 'Sales per customer', 'Late_delivery_risk',
       'Category Id', 'Category Name', 'Customer City', 'Customer Country',
       'Customer Id', 'Customer Segment', 'Customer State', 'Customer Zipcode',
       'Department Id', 'Department Name', 'Market', 'Order City',
       'Order Country', 'Order Customer Id', 'Order Id',
       'Order Item Cardprod Id', 'Order Item Discount',
       'Order Item Discount Rate', 'Order Item Id', 'Order Item Product Price',
       'Order Item Profit Ratio', 'Order Item Quantity', 'Sales',
       'Order Item Total', 'Order Profit Per Order', 'Order Region',
       'Order State', 'Order Status', 'Product Card Id', 'Product Category Id',
       'Product Name', 'Product Price', 'Shipping Mode', 'order_yr',
       'order_month', 'order_day', 'order_hour', 'late_days'


Variables Selected for Neural Network Model
      'Order Status', 'Shipping Mode', 'late_days'
Outcome
      'Late_delivery_risk'
Table 3.1. Data Table


