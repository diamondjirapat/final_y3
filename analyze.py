import pandas as pd

df = pd.read_csv('cleansing_water_data.csv')
print(df.shape)

# factor
factor_cols = [col for col in df.columns if col.startswith('factor_')]
for col in factor_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

top_factors = df[factor_cols].mean().sort_values(ascending=False)
print("\n--- ปัจจัยที่มีผลต่อการเลือกซื้อมากที่สุด (เรียงตามคะแนนเฉลี่ย) ---")
print(top_factors)

# Brand Primary
print("\n--- จำนวนผู้ใช้แต่ละแบรนด์เป็นหลัก (Top 5) ---")
print(df['brand_primary'].value_counts().head(5))

# Switch Factors
all_switch_factors = df['switch_factors'].dropna().str.split(', ').explode() #explode = แยกลูกน้ำ
print("\n--- เหตุผลหลักที่ทำให้ผู้บริโภคเปลี่ยนแบรนด์ (Top 5) ---")
print(all_switch_factors.value_counts().head(5))

# Core Value Proposition
if 'Kiyora' in df['brand_primary'].unique():
    kiyora_factors = df[df['brand_primary'] == 'Kiyora'][factor_cols].mean()
    others_factors = df[df['brand_primary'] != 'Kiyora'][factor_cols].mean()

    comparison = pd.DataFrame({'Kiyora': kiyora_factors, 'Others': others_factors})
    comparison['Difference'] = comparison['Kiyora'] - comparison['Others']
    
    print("\n--- เปรียบเทียบคะแนนปัจจัยการเลือกซื้อ: Kiyora vs แบรนด์อื่นๆ ---")
    print(comparison.sort_values('Difference', ascending=False))
    
    # Skin Concerns (Kiyora customer)
    print("\n--- ปัญหาผิวของคนที่ใช้ Kiyora (Top 5) ---")
    kiyora_concerns = df[df['brand_primary'] == 'Kiyora']['concerns'].dropna().str.split(', ').explode()
    print(kiyora_concerns.value_counts().head(5))


