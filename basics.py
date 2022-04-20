import matplotlib.pyplot as plt

# plt.plot(train_counter, train_losses, color='#2962FF')
plt.scatter([0,1,2,3,4,5],[2.3316362548828127, 2.301470703125, 1.3466606811523438, 0.6341075622558594, 0.43373695373535154, 0.4071222961425781], color='#2962FF')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

