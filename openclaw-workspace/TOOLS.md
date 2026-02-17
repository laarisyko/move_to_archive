# SSSI CLI Tool

The `sssi` command manages participation in the Super Safe Super Intelligence
decentralized LLM network.

## Quick Reference

```bash
sssi detect --json                    # Auto-detect GPU/CPU resources
sssi node start                       # Start local P2P node (Docker)
sssi node stop                        # Stop local P2P node
sssi node logs                        # View node logs
sssi join --gpu-memory 8GB --json     # Join network, advertise compute
sssi status --json                    # Check node health
sssi peers --json                     # List connected peers
sssi models --json                    # List available models
sssi rounds --json                    # List active training rounds
sssi infer -m MODEL -p "PROMPT" --json  # Run inference
sssi train -m MODEL -r 5 --json      # Join training rounds
sssi evolve -m MODEL --mutation TYPE --position N --json  # Propose mutation
sssi vote --proposal ID --decision approve --json         # Vote on proposal
```

## JSON Output

Always use `--json` when you need to parse the result. Every command returns
structured JSON when this flag is set.

## Mutation Types

For `sssi evolve --mutation`:
- `add_layer` -- Insert a new layer at position
- `remove_layer` -- Remove the layer at position
- `widen_layer` -- Increase the output dimension of a layer
- `swap_activation` -- Change the activation function
- `insert_skip` -- Add a skip/residual connection
