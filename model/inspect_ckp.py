if __name__ == '__main__':
    pwd = '/datadrive/GETSum/cnndm_disco_rst_43789/best.th'
    import torch

    x = torch.load(pwd)
    print(x)