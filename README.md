# E-Net  
**E-Net在业界因为速度快，深受人们好评，本库帮助大家了解掌握E-Net网络  
**1.本项目主要包含一下内容：**  
|————build:  
----------|————train:  
--------------------|————images  
--------------------|————labels  
----------|————test:  
--------------------|————images  
--------------------|————labels  
----------|————val:  
--------------------|————images  
--------------------|————labels  
|————losses:  
-----------|———— B_Focal_loss.py  用于二分类的Focal Loss 
-----------|———— C_Focal_loss.py  用于多分类（包括二分类）的Focal Loss 
-----------|———— Dice_loss.py     用于多分类（包括二分类）的Dice Loss 
-----------|———— BCE_Dice_loss.py 用于二分类的Dice Loss和二元交叉熵Loss加权 
-----------|———— CE_Dice_loss.py  用于多分类的Dice Loss和多元交叉熵Loss加权 
-----------|———— Tversky_loss.py  用于多分类（包括二分类）的Tversky Loss 
-----------|———— Focal_Tversky_loss.py 用于多分类（包含二分类）的Focal Loss和Tversky_loss加权 
-----------|———— Weighted_Categorical_loss.py 带权重的交叉熵损失，可以平衡原本数量 
-----------|———— Generalized_Dice_loss.py 改善Dice Loss，将多个类别的Dice Loss进行整合，使用一个参数作为分割结果的量化指标
-----------|———— Jaccard_Loss.py 实现了Jaccard Loss   
-----------|———— BCE_Jaccard_Loss.py 实现了BCE_Jaccard_Loss  
-----------|———— CE_Jaccard_Loss.py 实现了CE_Jaccard_Loss  
|————metrics：  
------------|————metrics  
|————output  
|————data
|————eval  
|————model
|————maxmin     
|————train     
|————new_train    
|————predict   
**2.包含多种损失函数：**    
|————losses:     
-----------|———— B_Focal_loss.py  用于二分类的Focal Loss 
-----------|———— C_Focal_loss.py  用于多分类（包括二分类）的Focal Loss     
-----------|———— Dice_loss.py     用于多分类（包括二分类）的Dice Loss    
-----------|———— BCE_Dice_loss.py 用于二分类的Dice Loss和二元交叉熵Loss加权     
-----------|———— CE_Dice_loss.py  用于多分类的Dice Loss和多元交叉熵Loss加权     
-----------|———— Tversky_loss.py  用于多分类（包括二分类）的Tversky Loss     
-----------|———— Focal_Tversky_loss.py 用于多分类（包含二分类）的Focal Loss和Tversky_loss加权 
-----------|———— Weighted_Categorical_loss.py 带权重的交叉熵损失，可以平衡原本数量        
-----------|———— Generalized_Dice_loss.py 改善Dice Loss，将多个类别的Dice Loss进行整合，使用一个参数作为分割结果的量化指标  
-----------|———— Jaccard_Loss.py 实现了Jaccard Loss     
-----------|———— BCE_Jaccard_Loss.py 实现了BCE_Jaccard_Loss   
-----------|———— CE_Jaccard_Loss.py 实现了CE_Jaccard_Loss   
**3.多种评价指标：**          
|————metrics：  
------------|————metrics      
**4.采用马赛诸塞州道路图数据集：**      









