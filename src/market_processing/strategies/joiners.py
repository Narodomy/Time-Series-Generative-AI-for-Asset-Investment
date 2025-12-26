from .base import AlignmentStrategy

class StrictAlignment(AlignmentStrategy):
    """Inner Join: เอาเฉพาะช่วงเวลาที่มีครบทุกตัว (ข้อมูลหาย แต่ Correlation เป๊ะ)"""
    def align(self, data_dict):
        # Implementation of pd.concat with join='inner'
        pass

class FillAlignment(AlignmentStrategy):
    """Forward Fill: ถมข้อมูล (ดีกับ Lead/Lack แต่ต้องระวัง Noise)"""
    def align(self, data_dict):
        # Implementation of pd.concat with ffill()
        pass