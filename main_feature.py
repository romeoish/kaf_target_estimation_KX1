import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# --- 0. 설정 및 상수 정의 ---

MODEL_DIR = 'model_reverse' 

# ⭐️ 역방향 모델의 입력 Feature 정의 (API의 입력값이 될 것)
inverse_features = [
    'Denier', 'Elongation', 'Tenacity', 'Cohesion', 'TotalFinish',  # 기존 물성값
    '시스SP속도', '코어SP속도', 'FeedRoll속도(M6)', '원료','Can수'  # 사용자가 입력할 고정값
]

# ⭐️ 역방향 모델의 출력 Target 정의 (API의 출력값이 될 것)
inverse_targets = [
    'DS-1연신비', "CrBox압력", "CrRoll압력", 'Bath온도', 'Steam분사',
    'Cutter속도', 'DS-3속도', 'Spray농도', 'Spray분사량'
]

# --- 1. 저장된 역방향 모델 및 LabelEncoder 로드 ---
loaded_inverse_models: Dict[str, Any] = {}
loaded_inverse_label_encoder_원료: Any = None 

print(f"Loading inverse models and LabelEncoders from: {MODEL_DIR}")

for target_name in inverse_targets:
    # 모델 파일명은 final_inverse_model_로 시작하고 특수문자는 언더스코어로 변경
    model_filename = os.path.join(MODEL_DIR, f'inverse_model_{target_name}.pkl')
    if os.path.exists(model_filename):
        try:
            loaded_inverse_models[target_name] = joblib.load(model_filename)
            print(f"✅ Loaded inverse model: {model_filename}")
        except Exception as e:
            print(f"❌ Failed to load inverse model {model_filename}: {e}")
    else:
        print(f"❌ Inverse model file not found: {model_filename}")

le_filename = os.path.join(MODEL_DIR, 'label_encoder_inverse_원료.joblib')
if os.path.exists(le_filename):
    try:
        loaded_inverse_label_encoder_원료 = joblib.load(le_filename)
        print(f"✅ Loaded '원료' LabelEncoder: {le_filename}")
    except Exception as e:
        print(f"❌ Failed to load '원료' LabelEncoder {le_filename}: {e}")
else:
    print(f"❌ '원료' LabelEncoder file not found: {le_filename}")
    print(f"⚠️ WARNING: '원료' LabelEncoder is crucial for inverse prediction if '원료' is an input feature. Please ensure it exists.")

# --- 2. FastAPI 애플리케이션 정의 ---

app = FastAPI(
    title="역방향 예측 API (업데이트 버전)",
    description="물성값과 고정 조건으로 나머지 조건값을 직접 예측하는 API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Pydantic 모델 정의 ---

class InverseInput(BaseModel):
    Denier: float = Field(..., description="원하는 Denier 값")
    Elongation: float = Field(..., description="원하는 Elongation 값")
    Tenacity: float = Field(..., description="원하는 Tenacity 값")
    Cohesion: float = Field(..., description="원하는 Cohesion 값")
    TotalFinish: float = Field(..., description="원하는 TotalFinish 값") 

    시스SP속도: float = Field(..., description="고정 시스SP속도")
    코어SP속도: float = Field(..., description="고정 코어SP속도")
    FeedRoll속도_M6: float = Field(..., alias="FeedRoll속도(M6)", description="고정 FeedRoll속도(M6)")
    원료: str = Field(..., description="고정 원료 종류")
    Can수: float = Field(..., description="고정 Can수")

    class Config:
        populate_by_name = True # Pydantic V2
        json_schema_extra = { 
            "example": {
                "Denier": 75.0,
                "Elongation": 30.0,
                "Tenacity": 35.0,
                "Cohesion": 15.0,
                "TotalFinish": 5.0, 
                "시스SP속도": 25.0,
                "코어SP속도": 1000.0,
                "FeedRoll속도(M6)": 850.0,
                "원료": "원료A",
                "Can수": 15.0
            }
        }

class InverseOutput(BaseModel):
    DS_1연신비: float = Field(..., alias="DS-1연신비", description="예측된 DS-1연신비")
    CrBox압력: float = Field(..., description="예측된 CrBox압력")
    CrRoll압력: float = Field(..., description="예측된 CrRoll압력")
    Bath온도: float = Field(..., description="예측된 Bath온도")
    Steam분사: float = Field(..., description="예측된 Steam분사")
    Cutter속도: float = Field(..., description="예측된 Cutter속도 (원본 스케일)")
    DS_3속도: float = Field(..., alias="DS-3속도", description="예측된 DS-3속도")
    Spray농도: float = Field(..., description="예측된 Spray농도")
    Spray분사량: float = Field(..., description="예측된 Spray분사량")

    class Config:
        populate_by_name = True 

# --- 4. 예측 함수 정의 ---
def predict_inverse_conditions_simplified(inputs: InverseInput) -> Dict[str, float]:
    
    input_data_dict_internal = inputs.model_dump(by_alias=False) 
    
    encoded_원료 = None
    if '원료' in inverse_features:
        if loaded_inverse_label_encoder_원료:
            try:
                encoded_원료 = loaded_inverse_label_encoder_원료.transform([input_data_dict_internal['원료']])[0]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"입력된 '원료' 값('{input_data_dict_internal['원료']}')은 학습된 LabelEncoder에 없습니다.")
        else:
            raise HTTPException(status_code=500, detail="'원료' LabelEncoder가 로드되지 않았습니다. 모델 로드 단계를 확인하세요.")

    # 예측에 사용할 입력 데이터프레임 생성
    input_df_data = {}
    for feature_col in inverse_features: 
        if feature_col == '원료':
            input_df_data[feature_col] = encoded_원료
        elif feature_col == 'TotalFinish': 
            input_df_data[feature_col] = input_data_dict_internal['TotalFinish'] 
        elif feature_col == '시스SP속도':
            input_df_data[feature_col] = input_data_dict_internal['시스SP속도']
        elif feature_col == '코어SP속도':
            input_df_data[feature_col] = input_data_dict_internal['코어SP속도']
        elif feature_col == 'FeedRoll속도(M6)':
            input_df_data[feature_col] = input_data_dict_internal['FeedRoll속도_M6']
        elif feature_col == 'Can수':
            input_df_data[feature_col] = input_data_dict_internal['Can수']
        else: # Denier, Elongation, Tenacity, Cohesion
            input_df_data[feature_col] = input_data_dict_internal[feature_col]

    input_df = pd.DataFrame([input_df_data])

    results = {}
    for target_name in inverse_targets:
        if target_name not in loaded_inverse_models:
            raise HTTPException(status_code=500, detail=f"'{target_name}'에 대한 역방향 모델이 로드되지 않았습니다.")
        
        model = loaded_inverse_models[target_name]
        try:
            y_pred = model.predict(input_df)[0]
            
            # Cutter속도는 로그 변환된 값을 예측하므로 지수 변환 필요
            if target_name == 'Cutter속도':
                y_pred = np.exp(y_pred)  # 로그 역변환
                print(f"Cutter속도 로그 역변환: 로그값={model.predict(input_df)[0]:.4f} -> 원본값={y_pred:.4f}")
            
            # 결과 딕셔너리에 추가 (Pydantic 필드명에 맞게)
            if target_name == 'DS-1연신비':
                results['DS_1연신비'] = round(float(y_pred), 4)
            elif target_name == 'DS-3속도':
                results['DS_3속도'] = round(float(y_pred), 4)
            else:
                # 나머지는 target_name 그대로 사용
                results[target_name] = round(float(y_pred), 4)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"'{target_name}' 모델 예측 중 오류 발생: {str(e)}")

    return results

# --- 5. API 엔드포인트 정의 ---

@app.post("/KS-2", response_model=InverseOutput, summary="물성값 및 고정 조건 기반 역방향 조건 예측")
async def direct_inverse_predict(inputs: InverseInput):
    """
    물성값(Denier, Elongation, Tenacity, Cohesion, TotalFinish)과 
    고정 조건값(시스SP속도, 코어SP속도, FeedRoll속도(M6), 원료, Can수)을 입력받아
    나머지 조건값들을 예측합니다.
    
    주의: Cutter속도는 로그 변환되어 학습되었으므로 자동으로 원본 스케일로 역변환됩니다.
    """
    try:
        result = predict_inverse_conditions_simplified(inputs)
        return InverseOutput(**result)
    except HTTPException as e:
        raise e 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 알 수 없는 오류 발생: {str(e)}")

@app.get("/", summary="API 상태 확인")
async def root():
    return {
        "message": "역방향 예측 API가 정상 작동 중입니다.",
        "loaded_models": list(loaded_inverse_models.keys()),
        "model_count": len(loaded_inverse_models),
        "features": inverse_features,
        "targets": inverse_targets
    }

@app.get("/health", summary="헬스 체크")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(loaded_inverse_models),
        "label_encoder_loaded": loaded_inverse_label_encoder_원료 is not None
    }

# --- 6. API 실행 (로컬 서버 구동) ---
if __name__ == "__main__":
    print(f"\nStarting FastAPI server on http://0.0.0.0:8003")
    print(f"Access interactive API documentation (Swagger UI) at http://0.0.0.0:8003/docs")
    print(f"Loaded models: {list(loaded_inverse_models.keys())}")
    print(f"Total models loaded: {len(loaded_inverse_models)}")
    uvicorn.run(app, host="0.0.0.0", port=8003)