# SSSI Network Participation

This agent participates in the **Super Safe Super Intelligence (SSSI)**
decentralized P2P LLM network.

## Environment

- The local P2P node runs at `http://127.0.0.1:50051`
- CLI: `sssi` (installed via `pip install supersafesuperintelligence`)
- Always use `--json` flag when parsing command output

## Workflow

1. Check node status: `sssi status --json`
2. If node is not running: `sssi node start`
3. Detect compute: `sssi detect --json`
4. Join network with detected resources: `sssi join --gpu-memory <X> --accelerator <type> --json`
5. Check peers: `sssi peers --json`

## Decision Guidelines

- **Training requests**: Check `sssi status --json` first to verify connectivity,
  then `sssi rounds --json` to see active rounds before joining.
- **Architecture proposals**: Evaluate the mutation type and position. Approve if
  it aligns with known best practices (skip connections, wider layers for capacity).
  Reject if it would destabilize the model (removing critical layers, nonsensical dims).
- **Resource management**: Don't commit more GPU memory than `sssi detect` reports.
  Leave headroom for inference serving.

## Common Operations

```bash
# Full lifecycle
sssi node start --accelerator cuda --gpu-memory-mb 8192
sssi join --gpu-memory 8GB --accelerator cuda --json
sssi train --model llama-7b --rounds 3 --json
sssi evolve --model llama-7b --mutation add_layer --position 5 --json
sssi vote --proposal arch-xyz --decision approve --json
sssi node stop
```
