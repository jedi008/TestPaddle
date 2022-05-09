import paddle
from paddle.static import InputSpec

input = InputSpec([None, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')

model = paddle.Model(paddle.vision.models.LeNet(),
    input, label)
optim = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=model.parameters())
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss())

params_info = model.summary()
print(params_info)