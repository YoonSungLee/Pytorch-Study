import torch

def train(trainloader, testloader, model, criterion, optimizer, num_epochs, save_interval):

    path = './savemodel/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available.')
    model = model.to(device)
    print('Start!')
    for epoch in range(num_epochs):

        running_loss = 0.0

        # enumerate(iterable, start=0)
        # iterable - a sequence, an iterator, or objects that supports iteration
        # start (optional) - enumerate() starts counting from this number. If start is omitted, 0 is taken as start.

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        cost = running_loss/len(trainloader)
        print('[%d] loss: %.3f' % (epoch + 1, cost))
        if epoch % save_interval == save_interval-1:
            print('Save model parameters - [%d] loss: %.3f' % (epoch + 1, cost))
            torch.save(model.state_dict(), path + 'trained_model'+str(epoch)+'.pth')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Train Accuracy: %d %%' % (100 * correct / total))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: %d %%' % (100 * correct / total))