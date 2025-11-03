import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
buffer = io.StringIO()
df = pd.read_csv(f"./train.csv")
df_test = pd.read_csv(f"./test.csv")
st.title("Loan Payback Prediction App")

code = ['''
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
''' , '''df.head()''' , '''df.tail()''' , '''df.info''' , '''df.describe()''' , '''df.shape''' , '''df.isna().sum()''' , '''df.columns''' , '''for col in df.select_dtypes(include="object").columns:
    df = pd.get_dummies(df) ''' , '''for col in df_test.select_dtypes(include="object").columns:
    df_test = pd.get_dummies(df_test) ''' , '''from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_cols = ['annual_income','loan_amount','debt_to_income_ratio','credit_score','interest_rate']
df[num_cols] = scaler.fit_transform(df[num_cols])
df = df.set_index('id') 

df_test[num_cols] = scaler.transform(df_test[num_cols])''' , '''df.duplicated().sum() ''' , ''' num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != "loan_paid_back"]  # ลบ target ออก

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)''' , '''from sklearn.model_selection import StratifiedKFold # ใช้การแบ่ง data แบบ KFold
import numpy as np
Y = df["loan_paid_back"] # เลือก label
X = df.drop(["loan_paid_back"], axis=1) # drop label ทิ้ง
X_test = df_test.drop("id", axis=1) # drop id ทิ้งเพื่อใช้ในการนำไป predict เพื่อส่งคำตอบ
oof_preds = np.zeros(len(Y))  # เก็บ prediction ของ train
test_preds = np.zeros(len(X_test))  # เก็บ prediction ของ test set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # สร้าง KFold ในการ split ออกมา 5 Group ''' , '''df["loan_paid_back"].value_counts() ''' ,
'''import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
# Loop Fold แต่ละ Group
for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y), 1):
    print(f"Fold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    # MODEL XGB
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.01,
        n_estimators=10000,
        max_depth=5
    )

    model.fit(X_train, y_train,)
    xgb_val_pred = model.predict_proba(X_val)[:, 1]
    xgb_test_pred = model.predict_proba(X_test)[:, 1]

    # Model Light GBM
    lgb_model = LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.01,
            num_leaves=31, # จำนวนสูงสุดของใบไม้ สูตร num_leaves <= 2^(max_depth)
            max_depth=5
        )
    lgb_model.fit(X_train, y_train)
    lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]

    # --- Combine predictions (ensemble) ---
    val_pred = 0.7 * xgb_val_pred + 0.3 * lgb_val_pred  # ตัวอย่าง weight XGB 70%, LGB 30%
    test_pred = 0.7 * xgb_test_pred + 0.3 * lgb_test_pred

    # เก็บ OOF prediction
    oof_preds[val_idx] = val_pred

    # เก็บ test prediction (เฉลี่ยทุก fold)
    test_preds += test_pred / skf.n_splits ''']
st.subheader("อ่านไฟล์ CSV ด้วย Pandas")
st.code(code[0] , language="python")

st.subheader("ดูข้อมูล 5 แถวแรก")
st.code(code[1] , language="python")
st.dataframe(df.head())

st.subheader("ดูข้อมูล 5 แถวท้าย")
st.code(code[2] , language="python")
st.dataframe(df.tail())


st.subheader("ดูinformation ข้อมูลเบื้องต้น")
st.code(code[3] , language="python")
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("ดูข้อมูลทางสถิติของข้อมูลชุดนี้")
st.code(code[4] , language="python")
st.dataframe(df.describe())

st.subheader("ดูจำนวน row และ columns")
st.code(code[5] , language="python")
shape_df = pd.DataFrame({
    "Rows": [df.shape[0]],
    "Columns": [df.shape[1]]
})
st.dataframe(shape_df)

st.subheader("หาค่าว่างในข้อมูล")
st.code(code[6] , language="python")
st.dataframe(df.isna().sum())

st.subheader("หา row ที่ซ้ำ")
st.code(code[11] , language="python")
duplicated= df.duplicated().sum()
st.write(f"จำนวน row ที่ซ้ำทั้งหมด : **{duplicated} rows**")


st.subheader("ดูว่ามี columns อะไรบ้าง")
st.code(code[7] , language="python")
st.dataframe(df.columns)

for col in df.select_dtypes(include="object").columns:
    df = pd.get_dummies(df) 
for col in df_test.select_dtypes(include="object").columns:
    df_test = pd.get_dummies(df_test) 
st.subheader("ทำ one hot encoding")
st.code(code[8] , language="python")
st.code(code[9] , language="python")
st.write("DF_TRAIN")
st.dataframe(df)
st.write("DF_TEST")
st.dataframe(df_test)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != "loan_paid_back"]  # ลบ target ออก

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper) # ถ้าเป็น outlier จะ set ให้เป็น lower หรือ upper

st.subheader("ทำการหา outlier")
st.code(code[12] , language="python")
fig, axes = plt.subplots(len(num_cols) - 1, 1, figsize=(8, 5*(len(num_cols) - 1)))

for i, col in enumerate(num_cols):
    if col == "id": continue
    sns.boxplot(x=df[col], ax=axes[i - 1])
    axes[i - 1].set_title(f"Boxplot of {col}")

plt.tight_layout()
st.pyplot(fig)


st.subheader("Correlation Heatmap ของ Numeric Features")
st.write("ถ้า 2 feature มี correlation สูงมาก เลือกใช้ตัวใดตัวหนึ่งก็พอ")
st.write("สีแดง/น้ำเงินเข้ม : strong correlation , สีอ่อน : weak correlation")
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# สร้าง correlation matrix
corr = df[num_cols].corr()
# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
ax.set_title("Correlation Matrix", fontsize=16)
st.pyplot(fig)


# MinMaxScaler (scale ให้อยู่ 0–1)
scaler = MinMaxScaler()
num_cols = ['annual_income','loan_amount','debt_to_income_ratio','credit_score','interest_rate']
df[num_cols] = scaler.fit_transform(df[num_cols])
df = df.set_index('id') 

df_test[num_cols] = scaler.transform(df_test[num_cols])
st.subheader("ทำการ Normalize ข้อมูลให้อยู่ระหว่าง 0 - 1")
st.code(code[10] , language="python")
st.dataframe(df)

st.subheader("ดูแล้วพบว่าข้อมูล Imbalance จำนวน Label ไม่ balance กัน")
st.code(code[14] , language="python")
st.dataframe(df["loan_paid_back"].value_counts())

st.subheader("เตรียมข้อมูลไปเทรน model")
st.code(code[13] , language="python")


st.subheader("เลือกใช้ XGBOOST MODEL กับ LGBM คู่กัน")
st.write("สอง model นี้คล้ายกันแต่ LGBM บางทีเห็น inside ข้อมูลได้ดีกว่าในบางจุด")
st.code(code[15] , language="python")

oof_predict = pd.read_csv("./oof_predictions.csv")
oof_preds = oof_predict["oof_pred"].values
Y = oof_predict["true_label"].values 
fpr, tpr, _ = roc_curve(Y, oof_preds) # ใช้เพื่อ คำนวณ True Positive Rate (TPR) และ False Positive Rate (FPR) สำหรับ threshold ต่าง ๆ
roc_auc = auc(fpr, tpr) # 0.5 → model ไม่ดีกว่าการสุ่ม (random guess) 1.0 → model perfect classification ยิ่งใกล้ 1 → model ยิ่งดีในการแยก class 0 และ 1
# สร้าง figure
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
ax.plot([0,1], [0,1], linestyle='--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

# แสดงใน Streamlit
st.pyplot(fig)


report = classification_report(Y, (oof_preds > 0.5).astype(int), output_dict=True)
st.subheader("Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

st.image("score.png", caption="Kaggle Score board", use_column_width=True)
st.markdown("[KAGGLE Competitions](https://www.kaggle.com/code/kapaopu/loanpredict)")
