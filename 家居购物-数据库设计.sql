-- 创建家具网购需要的数据库和表-- 创建 home_furnishing
-- 删除,一定要小心
DROP DATABASE IF EXISTS home_furnishing; 

-- 创建数据库
CREATE DATABASE home_furnishing; 


USE home_furnishing; 

DROP TABLE `member`
-- 创建会员表
CREATE TABLE `member`( 
`id` INT PRIMARY KEY AUTO_INCREMENT, 
`username` VARCHAR(32) NOT NULL UNIQUE, 
`password` VARCHAR(32) NOT NULL, 
`email` VARCHAR(64) 
)CHARSET utf8 ENGINE INNODB

-- 测试数据

INSERT INTO member(`username`,`password`,`email`) 
VALUES('admin',MD5('admin'),'hsp@hanshunping.net'); 

INSERT INTO member(`username`,`password`,`email`) 
VALUES('milan123',MD5('milan123'),'milan123@hanshunping.net'); 

SELECT * FROM member

SELECT `id`,`username`,`password`,`email` FROM `member`
 WHERE `username`='admin'; 
 
 INSERT INTO `member`(`username`,`password`,`email`) 
 VALUES('jack',MD5('jack'), 'jack@sohu.com');
 
 SELECT `id`,`username`,`password`,`email` FROM `member` 
 WHERE `username`='admin' AND `password`=MD5('admin')
 
 
 -- 创建家居表(表如何设计)
-- 设计furn表 家居表
 -- 老师说 需求-文档-界面 
 -- 技术细节
 -- 有时会看到 id int(11) ... 11 表示的显示宽度,存放的数据范围和int 配合zerofill
 --                int(2) .... 2 表示的显示宽度
 --                67890 => int(11) 00000067890
 --                67890 => int(2)  67890
 -- 创建表的时候，一定注意当前是DB
 -- 表如果是第一次写项目，表的字段可能会增加,修改，删除
 
 CREATE TABLE `furn`(
 `id` INT UNSIGNED PRIMARY KEY AUTO_INCREMENT, #id
 `name` VARCHAR(64) NOT NULL, #家居名
 `maker` VARCHAR(64) NOT NULL, #制造商
 `price` DECIMAL(11,2) NOT NULL , #价格 定点数
 `sales` INT UNSIGNED NOT NULL, #销量
 `stock` INT UNSIGNED NOT NULL, #库存
 `img_path` VARCHAR(256) NOT NULL #存放图片的路径
 )CHARSET utf8 ENGINE INNODB  
-- 测试数据, 参考 家居购物-数据库设计.sql 
INSERT INTO furn(`id` , `name` , `maker` , `price` , `sales` , `stock` , `img_path`) 
VALUES(NULL , '北欧风格小桌子' , '熊猫家居' , 180 , 666 , 7 , 'assets/images/product-image/6.jpg');
 
INSERT INTO furn(`id` , `name` , `maker` , `price` , `sales` , `stock` , `img_path`) 
VALUES(NULL , '简约风格小椅子' , '熊猫家居' , 180 , 666 , 7 , 'assets/images/product-image/4.jpg');
 
INSERT INTO furn(`id` , `name` , `maker` , `price` , `sales` , `stock` , `img_path`) 
VALUES(NULL , '典雅风格小台灯' , '蚂蚁家居' , 180 , 666 , 7 , 'assets/images/product-image/14.jpg');
 
INSERT INTO furn(`id` , `name` , `maker` , `price` , `sales` , `stock` , `img_path`) 
VALUES(NULL , '温馨风格盆景架' , '蚂蚁家居' , 180 , 666 , 7 , 'assets/images/product-image/16.jpg');


DELETE FROM `furn` WHERE id = 8

-- 修改...
UPDATE `furn` SET `name` = '北欧风格小桌子' , `maker` = '熊猫家居', `price` = 200 , 
`sales` = 112 , `stock` = 88 , `img_path` = 'assets/images/product-image/6.jpg' 
WHERE id = 1  


SELECT COUNT(*) FROM `furn` -- 11 -> 3

-- 模糊查询 Mysql基础
SELECT COUNT(*) FROM `furn` WHERE NAME LIKE '%%'


SELECT `id`, `name` , `maker`, `price`, `sales`, `stock`, 
                `img_path` imgPath FROM furn WHERE `name` LIKE '%沙发%' LIMIT 0, 3


-- 订单表 order 
-- 参考界面来写订单
-- 认真分析字段和字段类型
-- 老师说明: 每个字段, 使用not null 来约束
--  字段类型的设计, 应当和相关的表的字段有对应
-- 外键是否要给? 1. 需要[可以从db层保证数据的一致性 ]  
-- 2. 不需要[] [外键对效率有影响, 应当从程序的业务保证一致性]
-- 是否需要一个外键的约束? 
-- FOREIGN KEY(`member_id`) REFERENCES `member`(`id`) 

CREATE TABLE `order` (
`id` VARCHAR(64) PRIMARY KEY, -- 订单号
`create_time` DATETIME NOT NULL, -- 订单生成时间  
`price` DECIMAL(11,2) NOT NULL, -- 订单的金额 
`status` TINYINT NOT NULL, -- 状态 0 未发货 1 已发货 2 已结账
`member_id` INT NOT NULL -- 该订单对应的会员id
)CHARSET utf8 ENGINE INNODB

-- 创建订单项表
CREATE TABLE `order_item`(
id INT PRIMARY KEY AUTO_INCREMENT, -- 订单项的id
`name` VARCHAR(64) NOT NULL, -- 家居名
`price` DECIMAL(11,2) NOT NULL, -- 家居价格
`count` INT NOT NULL, -- 数量
`total_price` DECIMAL(11,2) NOT NULL, -- 订单项的总价
`order_id` VARCHAR(64) NOT NULL -- 对应的订单号
)CHARSET utf8 ENGINE INNODB

-- 编写一个添加order的sql
INSERT INTO `order`(`id`,`create_time`,`price`,`status`,`member_id`) 
VALUES('sn000001',NOW(),100,0,2) 

-- 编写一个添加orderitem的sql
-- 每写一行，自己要知道，自己在做什么?
INSERT INTO `order_item`(`id`,`name`,`price`,`count`,`total_price`,`order_id`) 
VALUES(NULL,'北欧小沙发',200,2,400,'sn00002') 






 
 