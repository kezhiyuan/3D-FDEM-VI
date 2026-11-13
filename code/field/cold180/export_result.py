from pathlib import Path
import argparse, json, time
HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu-granted', action='store_true')
    a = p.parse_args()
    if not (a.execute and a.gpu_granted):
        raise ValueError('New exclusive GPU allocation required')
    from validate_protocol import require_frozen_protocol, validate_training_inputs
    protocol = require_frozen_protocol()
    import torch
    import AEM_Variational_Bayes_3D_FD_v8 as core
    from cv_export import finish_cv
    cfg = core.Config(**protocol['config'])
    validate_training_inputs(cfg)
    started = time.perf_counter()
    core.configure_runtime(cfg)
    device = torch.device('cuda')
    out = Path(cfg.result_dir)
    if (out / 'final_export_completion_cv_v3.json').exists():
        raise FileExistsError('Preserve completed export')
    payload = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    assert payload['config'] == protocol['config']
    flow = core.MultiscaleFlow3D(cfg, device)
    flow.load_state_dict(payload['flow'])
    flow.set_active_coupling_layers(len(flow.layers))
    flow.eval()
    problem = core.Problem(cfg, device)
    finish_cv(flow, problem, cfg, out, payload['step'], vars(core))
    (out / 'separate_export_timing.json').write_text(json.dumps({'status': 'completed', 'whole_seconds': time.perf_counter() - started, 'scope': 'Separate1024/256/8 export including source-checkpoint/operator loading, streaming moments, quantiles/events and full responses; no new optimization'}, indent=2) + '\n', encoding='utf8')
if __name__ == '__main__':
    main()
