import argparse

import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm import tqdm, trange

from dataloader import load_sample, normalize_sample
from model import RectCNN, PaperCNN
from runs import RunManager
from utils import cm2str


def train(loader, model, optimizer, args):
    model.train()

    progress = tqdm(loader)
    for x, y in progress:
        x, y = x.to(args.device), y.to(args.device)
        p = model(x)
        loss = F.cross_entropy(p, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_postfix({'loss': '{:4.3f}'.format(loss)})


def evaluate(loader, model, args):
    loss = 0
    Y, Yhat = [], []
    n_correct, n_processed = 0, 0

    with torch.no_grad():
        model.eval()
        progress = tqdm(loader)
        for x, y in progress:
            Y.append(y)

            x, y = x.to(args.device), y.to(args.device)
            p = model(x)
            loss += F.cross_entropy(p, y)
            yhat = p.argmax(dim=1)

            Yhat.append(yhat.cpu())

            n_correct += (y == yhat).sum().item()
            n_processed += y.shape[0]

            logloss = loss.item() / n_processed
            accuracy = n_correct / n_processed
            metrics = {
                'loss': '{:4.3f}'.format(logloss),
                'acc': '{:4d}/{:4d} ({:%})'.format(n_correct, n_processed, accuracy)
            }
            progress.set_postfix(metrics)

    Y = torch.cat(Y).tolist()
    Yhat = torch.cat(Yhat).tolist()

    cm = confusion_matrix(Y, Yhat)
    report = classification_report(Y, Yhat, target_names=loader.dataset.classes)

    progress.write(cm2str(cm, loader.dataset.classes))
    progress.write(report)

    metrics = {'loss': logloss, 'acc': accuracy}
    return metrics


def test(args):
    run = RunManager(args, ignore=('device', 'evaluate', 'no_cuda'), main='model')
    print(run)

    if args.model == '1d-conv':
        model = RectCNN(282)
    else:
        model = PaperCNN()
    model = model.double()

    print("Loading: {}".format(run.ckpt('best')))
    checkpoint = torch.load(run.ckpt('best'))
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)

    test_dataset = DatasetFolder('data/test', load_sample, ('.npy',), transform=normalize_sample)
    print(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    test_metrics = evaluate(test_loader, model, args)
    run.writeResults(test_metrics)


def main(args):
    run = RunManager(args, ignore=('device', 'evaluate', 'no_cuda'), main='model')
    print(run)

    train_dataset = DatasetFolder('data/train', load_sample, ('.npy',), transform=normalize_sample)
    val_dataset = DatasetFolder('data/val', load_sample, ('.npy',), transform=normalize_sample)

    print(train_dataset)
    print(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8)

    if args.model == '1d-conv':
        model = RectCNN(282)
    else:
        model = PaperCNN()
    model = model.double().to(args.device)

    optimizer = SGD(model.parameters(), lr=1e-2)

    # evaluate(val_loader, model, args)
    best = 0
    progress = trange(1, args.epochs)
    for epoch in progress:
        progress.set_description('TRAIN [CurBestAcc={:.2%}]'.format(best))
        train(train_loader, model, optimizer, args)
        progress.set_description('EVAL [CurBestAcc={:.2%}]'.format(best))
        metrics = evaluate(val_loader, model, args)

        is_best = metrics['acc'] > best
        best = max(metrics['acc'], best)
        if is_best:
            run.save_checkpoint({
                'epoch': epoch,
                'params': vars(args),
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'metrics': metrics
            }, is_best)

        metrics.update({'epoch': epoch})
        run.pushLog(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GW Glitches Classification')
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-m', '--model', choices=('1d-conv', 'paper'), default='paper')
    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.set_defaults(no_cuda=False, evaluate=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cpu')
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    if args.evaluate:
        test(args)
    else:
        main(args)
