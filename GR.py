from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch
from spikingjelly.activation_based import functional, surrogate, neuron
from model3 import DVSGestureNet
import torch.nn.functional as F
import time
import datetime
from torch.cuda import amp

root_dir = './DVS128Gesture'
train_set = DVS128Gesture(root=root_dir, train=True, data_type='frame', frames_number=16, split_by='number')
test_set = DVS128Gesture(root=root_dir, train=False, data_type='frame', frames_number=16, split_by='number')

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True
)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=16,
    shuffle=True,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)


net = DVSGestureNet(channels=128, spiking_neuron=neuron.LIFNode, 
                                       surrogate_function=surrogate.ATan(), detach_reset=True)

functional.set_step_mode(net, 'm')
functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

net.to('cuda:9')


scaler = amp.GradScaler()


epochs = 300

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

for epoch in range(0, epochs):
    start_time = time.time()
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for frame, label in train_data_loader:
        optimizer.zero_grad()
        frame = frame.to('cuda:9')
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to('cuda:9')
        label_onehot = F.one_hot(label, 11).float()

        if scaler is not None:
            with amp.autocast():
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            out_fr = net(frame).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(net)

    train_time = time.time()
    train_speed = train_samples / (train_time - start_time)
    train_loss /= train_samples
    train_acc /= train_samples

    lr_scheduler.step()
    
    train_acc_hist.append(train_acc)
    train_loss_hist.append(train_loss)

    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in test_data_loader:
            frame = frame.to('cuda:9')
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to('cuda:9')
            label_onehot = F.one_hot(label, 11).float()
            out_fr = net(frame).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)
            
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    
    test_time = time.time()
    test_speed = test_samples / (test_time - train_time)
    test_loss /= test_samples
    test_acc /= test_samples
    
    test_acc_hist.append(test_acc)
    test_loss_hist.append(test_loss)
        
    print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
    print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    
with open('variable_data.txt', 'w') as file:
    # Write the data for each variable
    file.write('train_loss_hist = ' + str(train_loss_hist) + '\n')
    file.write('train_acc_hist = ' + str(train_acc_hist) + '\n')
    file.write('test_loss_hist = ' + str(test_loss_hist) + '\n')
    file.write('test_acc_hist = ' + str(test_acc_hist) + '\n')

print("Data has been stored in 'variable_data.txt'.")