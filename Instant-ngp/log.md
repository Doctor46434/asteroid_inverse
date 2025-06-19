| 实验ID | 数据集路径                                  | 数据集大小 | 多轨还是单轨     | 仿真还是实测 | 网络最终的结构         | 实验效果                         |
|:-------|:--------------------------------------------|:-----------|:-----------------|:-------------|:-----------------------|:---------------------------------|
| exp01  | ./dataset/sys_data/contact_ball/test01      | 150        | 五轨、平行       | 仿真         | 双relu                 | 不太好                           |
| exp02  | ./dataset/sys_data/contact_ball/test01      | 150        | 五轨、平行       | 仿真         | 双relu                 | 不错                             |
| exp03  | ./dataset/sys_data/contact_ball/test01      |            |                  |              |                        | 不行                             |
| exp04  | ./dataset/sys_data/contact_ball/test01      |            |                  |              |                        | 不行                             |
| exp05  | ./dataset/real_data/2024on_convert          | 37*5       | 五轨，平行小角度 | 实测         | 双relu                 | 二维效果不行，三维形状不错没细节 |
| exp06  | ./dataset/real_data/2024on_convert          | 37*5       | 五轨，平行小角度 | 实测         | 双relu                 | 二维效果不行，三维形状不错没细节 |
| exp07  | ./dataset/real_data/2024on_convert          | 37*5       | 五轨，平行小角度 | 实测         | 透明度relu + 1-exp(-x) | 二维效果不行，三维形状不错没细节 |
| exp08  | ./dataset/real_data/real_data_reg_2024On_13 | 37         | 一轨，平行视角   | 实测         | 透明度relu + 1-exp(-x) | 三维不行                         |
| exp09  |                                             |            |                  |              | 透明度relu + 1-exp(-x) | 三维不行                         |
| exp10  |                                             |            |                  |              |                        |                                  |
| exp11  |                                             | 30*5       | 多轨，平行视角   | 仿真球       | 全新的表面结构         | 暂时无法收敛                     |
| exp04  |                                             |            |                  |              |                        |                                  |