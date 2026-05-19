# ขั้นตอนที่ 0: Import ไลบรารีที่จำเป็น (Slide หน้า 14) 
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima, ADFTest 

# 1. Import ข้อมูล (Slide หน้า 15) 
df = pd.read_csv('year_sales.csv') 

# 2. แปลงคอลัมน์ Year ให้เป็น datetime (Slide หน้า 15) 
df['Year'] = pd.to_datetime(df['Year']) 

# 3. ตั้งค่าคอลัมน์ Year เป็น Index (Slide หน้า 15) 
df.set_index('Year', inplace=True) 

# 4. ตั้งค่าตัวทดสอบ ADF Test (Slide หน้า 16)
adf_test = ADFTest(alpha=0.05) 

# 5. ทดสอบว่าข้อมูลต้องทำ Differencing หรือไม่ (Slide หน้า 16)
# แก้ไขตัวสะกดคำว่า should_diff ให้ถูกต้อง
print("Should diff?:", adf_test.should_diff(df))

# 6-7. แบ่งข้อมูลเป็น Train dataset และ Test dataset โดยตัดที่ 80% (Slide หน้า 16) 
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size] 
test = df.iloc[train_size:] 

# 8. สร้างโมเดล ARIMA อัตโนมัติด้วยข้อมูล Train (Slide หน้า 16, 17) 
# แก้ไขพารามิเตอร์แรกจาก df ให้เป็น train เพื่อให้โมเดลเรียนรู้จากข้อมูลชุดฝึกสอนอย่างถูกต้อง
model = auto_arima(
    train, 
    start_p=0, d=1, start_q=0, 
    max_p=5, max_d=5, max_q=5, 
    start_P=0, D=1, start_Q=0, 
    max_P=5, max_D=5, max_Q=5, 
    m=12, seasonal=True, 
    error_action='warn', trace=True, 
    suppress_warnings=True, stepwise=True, 
    random_state=20, n_fits=50 
)

# 9. สรุปรายละเอียดผลลัพธ์ของโมเดล ARIMA (Slide หน้า 18) 
print(model.summary()) 

# 10. สร้างชุดข้อมูลพยากรณ์ให้มีความยาวเท่ากับข้อมูล Test (Slide หน้า 18) 
# แก้ไขตำแหน่งวงเล็บปิดของ pd.DataFrame ให้ครอบคลุมฟังก์ชัน predict ทั้งหมด
prediction = pd.DataFrame(model.predict(n_periods=len(test)), index=test.index) 
prediction.columns = ['predicted'] 

# 11. สร้างกราฟแสดงผลรวมเปรียบเทียบ Train, Test และ Predicted (Slide หน้า 18)
plt.figure(figsize=(12, 6))
# แก้ไขพารามิเตอร์ของ plt.plot ให้ดึงค่า Sales ออกมาแสดงในแกน Y อย่างถูกต้อง
plt.plot(train.index, train['Sales'], label='Train Data', color='blue')
plt.plot(test.index, test['Sales'], label='Test Data', color='orange')
plt.plot(prediction.index, prediction['predicted'], label='Predicted Data', color='green', linestyle='--')

plt.title('Time Series Analysis & Forecasting (Assignment Result)')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)

# บันทึกภาพกราฟออกมาเป็นไฟล์รูปภาพเพื่อนำไปส่งในระบบคู่กับโค้ด (Slide หน้า 19) 
plt.savefig('timeseries_assignment_result.png', bbox_inches='tight')
plt.show()