1. 
请先全方位的理解代码，然后完成metric的内容
Acc Pass@8
Avg Step – complete
Distance left towards the optimal steps
optimal step is given
Efficiency: Tokens used for successfully complete a task
有上面4个metric，optimal step 是跟任务匹配的是一个具体的数值，你要算出Avg step 和最有step的距离的差值，比如 Acg -  optimal/optimal
Efficiency: 是完成任务所花费的平均 token 数量 这个要统计influence的token数目 
2. 我们inference是使用的openrouter，请让其自动读取本地的key，