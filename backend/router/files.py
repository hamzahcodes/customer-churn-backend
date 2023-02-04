import os
import shutil
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from .. import schemas, database

router = APIRouter(
    tags=['File-Handling']
)

@router.post('/uploadfile')
def upload_file(email: str, file: UploadFile = File(...)):
    path = os.getcwd()
    x = path.replace("\\", "/")
    file_location = f"{x}/temp/{file.filename}" 

    dictionary = { "email" : email }
    query = database.collection.find_one(dictionary, {"_id": 0})
    # print(query)
    file_lst = query["files"]

    if file.filename not in file_lst:
        file_lst.append(file.filename)
        prev = {"email": email}
        nextt = {"$set": {"files" : file_lst}}
        database.collection.update_one(prev, nextt)

        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)    
    
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # considering abs value as in sklearn 
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    
    return col_corr

@router.get('/processfile')
def process_predict_file(filename: str):
    path = os.getcwd()
    x = path.replace("\\", "/")
    file_location = f"{x}/temp/{filename}"
    # return { "file_location" : file_location}

    df = pd.read_csv(file_location)
    for col in df.columns:
        if 'customer' in col or 'Customer' in col or 'ID' in col or 'id' in col or 'name' in col or 'Name' in col or 'RowNumber' in col:
            df.drop([col], axis = 1, inplace = True)

    na_variables = [ var for var in df.columns if df[var].isnull().mean() > 0 ]
    # print('null value containing column',na_variables)
    
    for col in na_variables:
        df[col] = df[col].fillna(df[col].mode()[0])

    cat_variables = [ var for var in df.columns if len(df[var].unique()) < 10]

    if(df['Churn'][0] != 1 and df['Churn'][0] != 0):
        dictionary = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0}
        df['Churn'] = df['Churn'].map(dictionary)

    cat_variables.remove('Churn')

    for col in cat_variables:
        ordinal_labels = df.groupby([col])['Churn'].mean().sort_values().index
        ordinal_labels2 = {k:i for i,k in enumerate(ordinal_labels, 0)}
        df[f'{col}_ordinal_variables'] = df[col].map(ordinal_labels2)
        df.drop([col], axis = 1, inplace = True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df.drop([col], axis = 1, inplace = True)

    X = df.drop(['Churn'], axis = 1)
    y = df['Churn']

    corr_features = correlation(X, 0.7)
    for col in corr_features:
        X.drop([col], axis = 1, inplace = True)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import mean_absolute_error

    from sklearn.model_selection import train_test_split
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    from sklearn.metrics import accuracy_score

    tree_model = DecisionTreeClassifier(random_state=1)
    tree_model.fit(train_X, train_y)
    tree_pred = tree_model.predict(val_X)
    errorT = mean_absolute_error(val_y, tree_pred)
    # print('Decision Tree')
    # print('Predicted Values(0/1) : ', tree_pred)
    # print('Mean Absolute Error : ', errorT)
    # print('Accuracy Score :', accuracy_score(val_y, tree_pred))


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(train_X, train_y)
    forest_pred = forest_model.predict(val_X)

    # forest_pred = forest_pred.map()
    errorF = mean_absolute_error(val_y, forest_pred)
    # print('Random Forest')
    # print('Predicted Values(0/1) : ', forest_pred)
    # print('Mean Absolute Error : ', errorF)
    # print('Accuracy Score :', accuracy_score(val_y, forest_pred))

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(train_X, train_y)
    
    log_pred = lr.predict(val_X)
    errorL = mean_absolute_error(val_y, log_pred)
    # print('Logistic Regression')
    # print('Probable Values(0 - 1) : ', log_pred)
    # print('Mean Absolute Error : ', errorL)
    # print('Accuracy Score :', accuracy_score(val_y, log_pred))

    item = schemas.Item(rows= X.shape[0],
                cols= X.shape[1],
                predt= tree_pred.tolist(),
                predf= forest_pred.tolist(), 
                predL= log_pred.tolist(),
                tree_error= errorT,
                forest_error= errorF,
                logistic_reg_error= errorL
                )
    json_compatible_item_data = jsonable_encoder(item)
    return JSONResponse(content=json_compatible_item_data)