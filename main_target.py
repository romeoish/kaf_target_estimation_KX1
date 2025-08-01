import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Type
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

##### ===================== KS-2 연신&방사 feature를 활용한 target 값 예측 =====================

# 수정된 targets: 모델 훈련 시 사용된 컬럼명과 일치시킴
targets = {
    'Denier': ['시스SP속도', '코어SP속도', 'FeedRoll속도(M6)', 'DS-1연신비'],
    'Elongation': ['시스SP속도', '코어SP속도', 'FeedRoll속도(M6)', 'DS-1연신비', '원료_encoded'], 
    'Tenacity': ['시스SP속도', '코어SP속도', 'FeedRoll속도(M6)', 'DS-1연신비', '원료_encoded'], 
    'Cohesion': ["CrBox압력", "CrRoll압력", 'Bath온도', 'Steam분사',
                 'Can수', 'Cutter속도', 'DS-1연신비'],
    'TotalFinish': ['DS-3속도', 'Spray농도', 'Spray분사량','시스SP속도', '코어SP속도', 'FeedRoll속도(M6)', 'DS-1연신비']
}


class FeatureInput_KX_1_Y(BaseModel):
    # ---------------- Denier, Elongation, Tenacity 공통 Feature ----------------
    # '시스S/P속도'와 '코어S/P속도'를 포함 (원래 SP_속도 하나였지만 분리)
    시스_S_P_속도: float = Field(..., alias="시스SP속도")
    코어_S_P_속도: float = Field(..., alias="코어SP속도")
    Feed_Roll_속도_M6: float = Field(..., alias="FeedRoll속도(M6)")
    DS_1_연신비: float = Field(..., alias="DS-1연신비")
    원료: str = Field(..., alias="원료") # Elongation, Tenacity에 필요

    # ---------------- Cohesion 전용 ----------------
    CR_Box_압력: float = Field(..., alias="CrBox압력") # 'Cr' 제거 반영
    CR_Roll_압력: float = Field(..., alias="CrRoll압력") # 'Cr' 제거 반영
    Bath_온도: float = Field(..., alias="Bath온도") # 'Bath온도'
    Steam_분사: float = Field(..., alias="Steam분사") # 'Steam분사'
    Can_수: int = Field(..., alias="Can수") # 'Can수', 데이터 타입 int로 변경 권장
    Cutter_속도: float = Field(..., alias="Cutter속도") # 'Cutter속도'

    # ---------------- Total Finish 전용 ----------------
    DS_3_속도: float = Field(..., alias="DS-3속도") # 'DS-3속도'
    Spray_농도: float = Field(..., alias="Spray농도") # 'Spray농도'
    Spray_분사량: float = Field(..., alias="Spray분사량") # 'Spray분사량'

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True # alias 대신 실제 필드 이름으로도 접근 가능하게 함


def predict_KX_1_Y(features: FeatureInput_KX_1_Y):
    # Pydantic 모델을 딕셔너리로 변환하고, 데이터프레임으로 만듭니다.
    # by_alias=True를 사용하여 JSON 요청에서 받은 컬럼 이름(alias)으로 매핑된 딕셔너리를 생성합니다.
    input_data = pd.DataFrame([features.dict(by_alias=True)])

    model_dir = "model" # 모델과 인코더 파일이 저장된 디렉토리
    results = {}

    # '원료' LabelEncoder를 미리 로드합니다.
    le_원료_path = os.path.join(model_dir, "label_encoder_원료.joblib")
    le_원료 = None # 초기화
    if os.path.exists(le_원료_path):
        le_원료 = joblib.load(le_원료_path)
    else:
        print(f"경고: '원료' LabelEncoder 파일이 없습니다: {le_원료_path}. '원료' 컬럼이 필요한 예측에서 문제가 발생할 수 있습니다.")

    for target, feature_cols in targets.items():
        model_path = os.path.join(model_dir, f"{target}_xgb_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"🔴 모델 파일이 없습니다: {model_path}")

        model = joblib.load(model_path)

        # 🎯 예측에 필요한 feature만 추출
        input_features = input_data.copy()

        # ❗ 범주형 처리: '원료_encoded' 컬럼이 현재 target의 feature_cols에 포함되어 있는지 확인
        if "원료_encoded" in feature_cols:
            if "원료" in input_features.columns:
                if le_원료 is not None:
                    # '원료' 컬럼을 인코딩하고 '원료_encoded'로 저장
                    try:
                        input_features["원료_encoded"] = le_원료.transform(input_features["원료"].astype(str))
                    except ValueError as e:
                        # 알려지지 않은 카테고리가 있을 경우 처리
                        print(f"경고: 알려지지 않은 '원료' 값이 있습니다: {e}")
                        # 기본값으로 첫 번째 클래스를 사용하거나 에러를 발생시킬 수 있습니다
                        input_features["원료_encoded"] = 0  # 또는 적절한 기본값
                    
                    # 원본 '원료' 컬럼 제거 (모델이 예상하지 않으므로)
                    input_features = input_features.drop(columns=["원료"])
                else:
                    raise RuntimeError("🔴 '원료' LabelEncoder가 로드되지 않아 '원료' 컬럼을 처리할 수 없습니다.")
            else:
                raise ValueError("🔴 '원료_encoded' 컬럼이 targets에 지정되었으나, 입력 데이터에 '원료' 컬럼이 없습니다.")

        # 필요한 피처만 선택
        try:
            input_features_selected = input_features[feature_cols]
        except KeyError as e:
            missing_cols = [col for col in feature_cols if col not in input_features.columns]
            raise ValueError(f"🔴 필요한 컬럼이 누락되었습니다: {missing_cols}")

        # 예측 수행
        y_pred = model.predict(input_features_selected)[0]
        results[target] = round(float(y_pred), 4)

    return {"prediction": results}


# ✅ CORS 설정 포함 FastAPI 앱 정의
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]
app = FastAPI(middleware=middleware)


# ✅ API 엔드포인트
@app.post("/KX-1")
async def predict_target(features: FeatureInput_KX_1_Y):
    try:
        result = predict_KX_1_Y(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {str(e)}")

# 로컬 서버 구동
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)