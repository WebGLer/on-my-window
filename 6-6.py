from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import preprocessing
path  = r"F:\5-model data\imdb.npz"
max_feature = 10000
maxlen = 20
(x_trian,y_train),(x_test,y_test) = imdb.load_data(
    path=path,
    maxlen=maxlen
)
# print(x_trian)
# '''
# [list([1, 6, 675, 7, 300, 127, 24, 895, 8, 2623, 89, 753, 2279, 5, 6866, 78, 14, 20, 9])
#  list([1, 11300, 390, 1351, 9, 4, 118, 390, 7, 11300, 45, 61, 514, 390, 7, 11300])
#  list([1, 13, 586, 851, 14, 31, 60, 23, 2863, 2364, 314])
#  list([1, 14, 20, 9, 394, 21, 12, 47, 49, 52, 302])
#  list([1, 13, 92, 124, 138, 13, 40, 14, 20, 38, 73, 21, 13, 115, 79, 1458, 7, 149, 12])
#  list([1, 1390, 128, 2257, 723, 8965, 60, 48, 25, 28, 296, 12])
#  list([1, 196, 357, 16445, 115, 28, 13, 77, 38, 1264, 8, 67, 277, 898, 1686])
#  list([1, 12039, 4, 12632, 127, 6, 117, 2, 5, 6, 20, 91, 3939])
#  list([1, 14, 9, 4, 6279, 20, 310, 7, 3420, 3394, 1902, 164, 21, 50, 26, 57, 1053, 388])
#  list([1, 11300, 390, 1351, 9, 4, 118, 390, 7, 11300, 45, 61, 514, 390, 7, 11300])
#  list([1, 931, 14, 20, 9, 1167, 9, 394, 55, 6415, 78, 2956, 963, 458, 24, 168])
#  list([1, 6741, 20576, 9, 321, 9, 14, 22, 29, 166, 6, 1429, 255])
#  list([1, 57, 931, 379, 20, 116, 856, 42, 433, 881, 57, 281, 33, 32, 1771, 12])
#  list([1, 53, 2570, 53, 1302, 76, 76, 53, 1193])
#  list([1, 530, 5, 728, 354, 34, 827, 5, 826, 10107])
#  list([1, 13, 440, 14, 604, 7, 22, 1188, 115, 796, 31571])
#  list([1, 332, 4, 274, 859, 4, 20])
#  list([1, 14, 9, 6, 87, 20, 99, 78, 12, 9, 24, 1439, 23, 344, 374])
#  list([1, 427, 777, 845, 13, 135, 586, 81, 14, 2184, 20, 4, 1351, 12, 1015, 106, 12, 150, 777])
#  list([1, 209, 6, 824, 31, 7, 8963, 118, 1711, 87, 318, 302, 5, 4, 11881, 72, 896])
#  list([1, 763, 14, 117, 1528, 8, 129, 1029, 7, 3182, 11147, 12, 9, 10, 10, 1047, 163, 5, 3308])
#  list([1, 43, 119, 4, 8466, 200, 107, 87, 105, 7, 868, 268, 12502, 5123])
#  list([1, 4101, 114, 4101, 458, 338, 2956])
#  list([1, 6, 6070, 22, 15, 434, 941, 129, 692, 1022, 5805, 9, 1429, 5, 8224, 8, 106])
#  list([1, 51, 6, 229, 51, 6, 65, 51, 6, 947])]
# '''
# print('*'*100)
# print(y_train)  #[0 1 0 0 1 0 0 0 1 1 0 1 0 0 1 0 0 1 1 1 1 1 0 1 0]
# print('*'*100)
# print(x_test)   #[]
# print('*'*100)
# print(y_test)   #[]
x_trian = preprocessing.sequence.pad_sequences(x_trian,maxlen=maxlen)
# print(x_trian)
'''
[[    0     1     6   675     7   300   127    24   895     8  2623    89
    753  2279     5  6866    78    14    20     9]
 [    0     0     0     0     1 11300   390  1351     9     4   118   390
      7 11300    45    61   514   390     7 11300]
 [    0     0     0     0     0     0     0     0     0     1    13   586
    851    14    31    60    23  2863  2364   314]
 [    0     0     0     0     0     0     0     0     0     1    14    20
      9   394    21    12    47    49    52   302]
 [    0     1    13    92   124   138    13    40    14    20    38    73
     21    13   115    79  1458     7   149    12]
 [    0     0     0     0     0     0     0     0     1  1390   128  2257
    723  8965    60    48    25    28   296    12]
 [    0     0     0     0     0     1   196   357 16445   115    28    13
     77    38  1264     8    67   277   898  1686]
 [    0     0     0     0     0     0     0     1 12039     4 12632   127
      6   117     2     5     6    20    91  3939]
 [    0     0     1    14     9     4  6279    20   310     7  3420  3394
   1902   164    21    50    26    57  1053   388]
 [    0     0     0     0     1 11300   390  1351     9     4   118   390
      7 11300    45    61   514   390     7 11300]
 [    0     0     0     0     1   931    14    20     9  1167     9   394
     55  6415    78  2956   963   458    24   168]
 [    0     0     0     0     0     0     0     1  6741 20576     9   321
      9    14    22    29   166     6  1429   255]
 [    0     0     0     0     1    57   931   379    20   116   856    42
    433   881    57   281    33    32  1771    12]
 [    0     0     0     0     0     0     0     0     0     0     0     1
     53  2570    53  1302    76    76    53  1193]
 [    0     0     0     0     0     0     0     0     0     0     1   530
      5   728   354    34   827     5   826 10107]
 [    0     0     0     0     0     0     0     0     0     1    13   440
     14   604     7    22  1188   115   796 31571]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     1   332     4   274   859     4    20]
 [    0     0     0     0     0     1    14     9     6    87    20    99
     78    12     9    24  1439    23   344   374]
 [    0     1   427   777   845    13   135   586    81    14  2184    20
      4  1351    12  1015   106    12   150   777]
 [    0     0     0     1   209     6   824    31     7  8963   118  1711
     87   318   302     5     4 11881    72   896]
 [    0     1   763    14   117  1528     8   129  1029     7  3182 11147
     12     9    10    10  1047   163     5  3308]
 [    0     0     0     0     0     0     1    43   119     4  8466   200
    107    87   105     7   868   268 12502  5123]
 [    0     0     0     0     0     0     0     0     0     0     0     0
      0     1  4101   114  4101   458   338  2956]
 [    0     0     0     1     6  6070    22    15   434   941   129   692
   1022  5805     9  1429     5  8224     8   106]
 [    0     0     0     0     0     0     0     0     0     0     1    51
      6   229    51     6    65    51     6   947]]
'''
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
print(x_test)