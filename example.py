print("Hello, World!")
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import  accuracy_score,precision_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn
import joblib
import boto3
import pathlib
from io import  StringIO
import argparse
import os
import numpy as np
import pandas as pd

def model(model_dir):
	clf=joblib.load(os.path.join(model,'model.joblib'))

if __name__=="__main__":
	print('[Info] Extrating argements')
	perser=argparse.ArgumentParser()

	perser.add_argument('--n_estimators',type=int,default=100)
	perser.add_argument('--max_depth',type=int,default=6)

	## data,model, output dir
	perser.add_argument('--model',type=str,default=os.environ.get('sSM_MODEL_DIR'))
	perser.add_argument('--train',type=str,default=os.environ.get('SM_CHANNEL_TRAIN'))
	perser.add_argument('--test',type=str,default=os.environ.get('SM_CHANNEL_TEST'))
	perser.add_argument('--train_file',type=str,default=os.environ.get('train-v1.csv'))
	perser.add_argument('--test_file',type=str,default=os.environ.get('test-v1.csv'))

	arg,_=perser.parse_known_args()

	print('sklearn version',sklearn.__version__)
	print('joblib version', joblib.__version__)

	print('readinf data')
	print()
	train_df=pd.read_csv(os.path.join(arg.train,arg.train_file))
	test_df=pd.read_csv(os.path.join(arg.test,arg.test_file))

	x=train_df.drop('status',axis=1)
	y=train_df['status']

	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.26,random_state=49)	
	
	print('Traning in aws sagemaker .....')
	model=RandomForestClassifier(n_estimators=arg.n_estimators,max_depth=arg.max_depth,verbose=3,n_jobs=-1)

	model.fit(x_train,y_train)

	print('fit data complete')

	model_path=os.path.join(arg.model,'model.joblib')
	joblib.dump(model,model_path)

	print(f' model saved in {model_path}')
	y_test_pred =model.predict(x_test)
	test_model_score = accuracy_score(y_test,y_test_pred)*100
	print('Model Reports .....')
	print(f"Training {model} accuracy {test_model_score}")

	print('precision score',precision_score(y_test,y_test_pred)*100)
	print('clf reports',classification_report((y_test,y_test_pred)))
