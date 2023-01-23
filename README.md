# smartathon
We have built a complete web app with a dedicated frontend, backend based on streamlit and an ml model made from the yolov7 architecture which has been deployed on AWS. 

Data preprocessing was first done to make train.csv compatible with yolo. We converted from the cab format to txt format. 

Multiple issues were encountered during data preprocessing and mapping of the CSV file and image file. Such as multiple annotations being present for the same image file, for multiple objects present in the image. Custom preprocessing functions were written to deal with this issue. As validation images didn't have any annotations present, we had to seperate those images from the train dataset. 
We also had to deal with the imbalance present in the dataset for the various image classes. We used image augmentation to deal with which. 

During training we encountered issues when it came training with such high resolution images, so we had to reduce the size to 320. 

After which we trained our model for as many epochs as we could with our present resources. If given the opportunity we belive we can deliver even better results.

https://user-images.githubusercontent.com/62798405/213971369-d5a6a5eb-e8a1-4817-9f4c-eec652084bef.mp4

