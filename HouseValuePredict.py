from pyspark.sql import SparkSession

# spark程序的第一步都是先创建SparkSession
spark = SparkSession.builder
        .master("local")
        .appName("hsy cf_housing")
        .config("spark.executor.memory", "1gb")
        .getOrCreate()
    
sc = spark.sparkContext

#读取并创建RDD

rdd = sc.textFile('/home/huang/ComputerApplying/cf_housing/CaliforniaHousing/cal_housing.data')

#读取数据每个属性的定义并创建RDD

header = sc.textFile('/home/huang/ComputerApplying/cf_housing/CaliforniaHousing/cal_housing.domain')

# 利用RDD的collect函数输出数据的属性

header.collect()

# 由于RDD中的数据比较大，所以接下来取前两个数据即可

rdd.take(2)

# 因为用的SparkContext的textFile函数去创建RDD，所以每个数据其实是一个大的字符串
# ，各个属性之间用逗号分隔开，所以为了方便后续操作，以下用map函数把大字符串分割成数组

rdd = rdd.map(lambda myline: myline.split(" ") )

rdd.take(2)#此时再取两个数据，命令行输出的结果就已经是数组形式的了


#因为DataFrame比rdd更方便性能也更加好，所以为了方便后续操作，以下用map函数把大字符串分割成数组
# 以下将rdd转化为DataFrame：
# 就是把之前用数据代表的rdd对象转换成为Row对象，再用toDF()函数转换成DataFrame

from pyspark.sql import Row

df = rdd.map(lambda line: Row(longitude=line[0],
                             latitude=line[1],
                             housingMedianAge=line[2],
                             totalRooms=line[3],
                             totalBedRooms=line[4],
                             population=line[5],
                             households=line[6],
                             medianIncome=line[7],
                             medianHouseValue=line[8]) ).toDF()
                             

# 此时可以用df.show()函数打印出这个DataFrame所包含的数据表

df.show()


# 此时show出来的每一列的数据格式都是string格式的，但是，它们其实都是数字，所以我们
# 可以通过cast()函数把每一列的类型转换成float类型的

def convertColumn(df, names, newType)
    for name in names:
        df = df.withColumn(name, df[name].cast(newType ) )
    return df
    

columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome', 'population', 'totalBedRooms', 'totalRooms']

df = convertColumn(df, columns, FloatType() )

#数据预处理
# 一、房价的值普遍偏大，我们在这里把它调整为相对较小的数字
# -->用withColumn()函数把所有房价都除以100000

#标注：withColumn()函数用于新建、调整列

df = df.withColumn("medianHouseValue", col("medianHouseValue")/100000 )

# 二、添加新的更有意义的列
# 1、每个家庭的平均房间数 roomsPerHousehold
# 2、每个家庭的平均人数    populationPerHousehold
#3、卧室在总房间的占比    bedroomsPerRoom

df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households"))
        .withColumn("populationPerHousehold", col("population")/col("households"))
        .withColumn("bedroomsPerRoom", col("totalBedRooms")/col("totalRooms"))
        

#舍弃经纬度，只选择重要的信息

df = df.select("medianHouseValue",
             "totalBedRooms",
             "population",
             "households",
             "medianIncome",
             "roomsPerHousehold",
             "populationPerHousehold",
             "bedroomsPerRoom")


# 三、要把房价和其他属性分开，即一个是labels，其他是features；所以可以把DataFrame转换成RDD，
# 然后用map()函数把每个对象分成两部分：房价和一个包含其余属性的列表，然后再转换回DataFrame
from pyspark.ml.linalg import DenseVector

input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:] ) ) )

df = spark.createDataFrame(input_data, ["label", "features"])

# 四、有的属性最小值和最大值之间范围很大，我们可以用SparkML库中的StandardScaler包来解决

# 输入是features，这里新增了一列输出,名叫features_scaled，它的每个数据都是标准化过的，我们可以用SparkML库中的StandardScaler包来解决
#我们应该用它而不是features来训练模型
standardScaler = standardScaler( inputCol = "features", outputCol = "features_scaled")

scaler = standardScaler.fit(df)

scaled_df = scaler.transform(df)


#创建线性回归模型

# 首先把数据集分为训练集和测试集，训练集用来训练模型，测试集用来评估模型的正确性
# 这里使用了DataFrame的randomSplit()函数来随机分割数据

train_data, test_data = scaled_df.randomSplit( [.8, .2], seed = 123 )

# 使用Spark ML的LinearRegression包

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features_scaled', labelCol = "label", maxIter = 10, regPlinearModel = lr.fit(train_data) )

#LinearRegression官方文档（如果想调整其他参数的话）：https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression


#模型评估阶段
# -->使用linearModel的transform()函数来预测测试集中的房价，
# 并与真实情况对比

predicted = linearModel.transform(test_data)

predictions = predicted.select("prediction").rdd.map()

labels = predicted.select("label").rdd.map(lambda x: x[0])

predictionAndLabel =  predictions.zip(labels).collect()

#取出两个结果来看看
predictionAndLabel[:2]



























