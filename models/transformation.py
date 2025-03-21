import torch
import pdb

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)



def smooth_ln_fcs_temporary(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    # scales 형태 확인 및 재형성
    if scales.numel() == 1:
        scales = scales.item() * torch.ones(ln.weight.size(0), device=scales.device)
    
    # shifts 형태 확인 및 재형성
    if shifts.numel() == 1:
        shifts = shifts.item() * torch.ones(ln.weight.size(0), device=shifts.device)
    
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        
        # 형태 확인 및 재형성
        if shifts.dim() == 1 and shifts.size(0) == fc.weight.size(1):
            temp_shifts = shifts
        else:
            print(f"Warning: fc shifts shape mismatch. shifts: {shifts.shape}, fc.weight: {fc.weight.shape}")
            if shifts.numel() == 1:
                temp_shifts = shifts.item() * torch.ones(fc.weight.size(1), device=shifts.device)
            else:
                # 기본값으로 대체
                temp_shifts = torch.zeros(fc.weight.size(1), device=shifts.device)
                
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@temp_shifts
        else:
            fc.temp_bias = fc.weight@temp_shifts
        fc.temp_weight = fc.weight * scales.view(1,-1)


def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = fc1.temp_bias - shifts
        fc1.temp_bias = fc1.temp_bias/scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/scales.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/scales.view(-1)
        fc1.temp_weight = fc1.weight/scales.view(-1,1)
    
    # 크기 일치 확인 및 재형성
    if shifts is not None:
        # shifts의 형태를 확인하고 필요시 재형성
        if shifts.dim() == 1 and shifts.size(0) == fc2.weight.size(1):
            pass  # 이미 올바른 형태
        elif shifts.dim() == 2 and shifts.size(0) == fc2.weight.size(1) and shifts.size(1) == 1:
            shifts = shifts.view(-1)  # 열 벡터를 1D 텐서로 변환
        elif shifts.dim() == 2 and shifts.size(1) == fc2.weight.size(1) and shifts.size(0) == 1:
            shifts = shifts.view(-1)  # 행 벡터를 1D 텐서로 변환
        elif shifts.numel() == 1:
            # 스칼라 값을 벡터로 확장
            shifts = shifts.item() * torch.ones(fc2.weight.size(1), device=shifts.device)
        else:
            # 차원 불일치 경고 출력
            print(f"Warning: shifts shape mismatch. shifts: {shifts.shape}, fc2.weight: {fc2.weight.shape}")
            # 기본값으로 대체
            shifts = torch.zeros(fc2.weight.size(1), device=shifts.device)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@shifts
    else:
        fc2.temp_bias = fc2.weight@shifts
    fc2.temp_weight = fc2.weight * scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)
    k_proj.temp_weight = k_proj.temp_weight*scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*scales.view(-1)

def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    # scales 형태 확인 및 재형성
    if scales.numel() == 1:
        scales = scales.item() * torch.ones(ln.weight.size(0), device=scales.device)
    
    # shifts 형태 확인 및 재형성
    if shifts.numel() == 1:
        shifts = shifts.item() * torch.ones(ln.weight.size(0), device=shifts.device)
        
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        
        # 형태 확인 및 재형성
        if shifts.dim() == 1 and shifts.size(0) == fc.weight.size(1):
            temp_shifts = shifts
        else:
            print(f"Warning: fc shifts shape mismatch. shifts: {shifts.shape}, fc.weight: {fc.weight.shape}")
            if shifts.numel() == 1:
                temp_shifts = shifts.item() * torch.ones(fc.weight.size(1), device=shifts.device)
            else:
                # 기본값으로 대체
                temp_shifts = torch.zeros(fc.weight.size(1), device=shifts.device)
            
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@temp_shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@temp_shifts)
        fc.weight.mul_(scales.view(1,-1))


def smooth_fc_fc_inplace(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False
    
    # 크기 일치 확인 및 재형성
    if shifts is not None:
        # shifts의 형태를 확인하고 필요시 재형성
        if shifts.dim() == 1 and shifts.size(0) == fc2.weight.size(1):
            pass  # 이미 올바른 형태
        elif shifts.dim() == 2 and shifts.size(0) == fc2.weight.size(1) and shifts.size(1) == 1:
            shifts = shifts.view(-1)  # 열 벡터를 1D 텐서로 변환
        elif shifts.dim() == 2 and shifts.size(1) == fc2.weight.size(1) and shifts.size(0) == 1:
            shifts = shifts.view(-1)  # 행 벡터를 1D 텐서로 변환
        elif shifts.numel() == 1:
            # 스칼라 값을 벡터로 확장
            shifts = shifts.item() * torch.ones(fc2.weight.size(1), device=shifts.device)
        else:
            # 차원 불일치 경고 출력
            print(f"Warning: shifts shape mismatch. shifts: {shifts.shape}, fc2.weight: {fc2.weight.shape}")
            # 기본값으로 대체
            shifts = torch.zeros(fc2.weight.size(1), device=shifts.device)
            
    fc1.bias.sub_(shifts)
    fc1.bias.div_(scales.view(-1))
    fc1.weight.div_(scales.view(-1,1))
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
    fc2.weight.mul_(scales.view(1,-1))

def smooth_q_k_inplace(q_proj, k_proj, scales,):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    q_proj.weight.div_(scales.view(-1,1))
    q_proj.bias.div_(scales.view(-1))
    k_proj.weight.mul_(scales.view(-1,1))
    k_proj.bias.mul_(scales.view(-1))