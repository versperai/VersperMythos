import React, { useMemo, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';

import type { BackendState } from '../hooks/useBackendSession';

export function Inference({
  session,
}: {
  session: BackendState & {
    sendRequest: (req: any) => void;
    clearGeneration: () => void;
  };
}): React.JSX.Element {
  // Derive available weights from discovered model list
  const availableModels = useMemo(() => {
    const list: Array<{ type: string; name: string; path: string }> = [];
    for (const m of session.modelList ?? []) {
      list.push({
        type: (m as any).type ?? 'lm',
        name: (m as any).name ?? '',
        path: (m as any).path ?? '',
      });
    }
    return list;
  }, [session.modelList]);

  // Pick the best default: prefer HF-format models (directories) over .pth checkpoints
  const defaultWeight = useMemo(() => {
    // Prefer HF models (name doesn't end with .pth, e.g. "lm/sft")
    const hf = availableModels.find((m) => m.type === 'lm' && !m.name.endsWith('.pth'));
    if (hf) return hf.name.replace('.pth', '');
    // Fallback to any LM model
    const lm = availableModels.find((m) => m.type === 'lm');
    if (lm) return lm.name.replace('.pth', '');
    // Fallback: first available model
    if (availableModels.length > 0) return availableModels[0].name.replace('.pth', '');
    return '';
  }, [availableModels]);

  const [prompt, setPrompt] = useState('');
  const [modelType, setModelType] = useState<'lm' | 'vlm' | 'rlm'>('lm');
  const [weight, setWeight] = useState(defaultWeight);
  const [maxTokens, setMaxTokens] = useState('512');
  const [temperature, setTemperature] = useState('0.7');
  const [inputMode, setInputMode] = useState<'prompt' | 'model' | 'weight' | 'tokens' | 'temp'>(
    'prompt',
  );
  const [loaded, setLoaded] = useState(false);

  // Filter available weights for current model type
  const weightCandidates = useMemo(
    () => availableModels.filter((m) => m.type === modelType).map((m) => m.name.replace('.pth', '')),
    [availableModels, modelType],
  );
  const [weightIdx, setWeightIdx] = useState(0);

  // Update default weight when models discovered or model type changes
  const effectiveWeight = weight || weightCandidates[0] || '';

  const handleLoadModel = (w: string): void => {
    session.clearGeneration();
    setLoaded(true);
    session.sendRequest({
      type: 'load_model',
      model_type: modelType,
      weight: w || effectiveWeight,
    });
  };

  const handleSubmit = (value: string): void => {
    if (!value.trim()) return;
    session.clearGeneration();
    session.sendRequest({
      type: 'generate',
      model_type: modelType,
      weight: effectiveWeight,
      prompt: value,
      max_new_tokens: parseInt(maxTokens, 10) || 512,
      temperature: parseFloat(temperature) || 0.7,
    });
    setPrompt('');
  };

  useInput((chunk, key) => {
    // 'l' / 'L' → load model (only when not typing in prompt)
    if ((chunk === 'l' || chunk === 'L') && inputMode === 'prompt') {
      handleLoadModel(effectiveWeight);
      return;
    }
    // Tab: cycle input modes
    if (key.tab && !key.shift) {
      const modes: Array<'prompt' | 'model' | 'weight' | 'tokens' | 'temp'> = [
        'prompt',
        'model',
        'weight',
        'tokens',
        'temp',
      ];
      setInputMode((m) => {
        const idx = modes.indexOf(m);
        return modes[(idx + 1) % modes.length];
      });
      return;
    }
    // In weight mode: up/down to cycle candidates
    if (inputMode === 'weight' && weightCandidates.length > 0) {
      if (key.upArrow) {
        setWeightIdx((i) => Math.max(0, i - 1));
        return;
      }
      if (key.downArrow) {
        setWeightIdx((i) => Math.min(weightCandidates.length - 1, i + 1));
        return;
      }
      if (key.return && inputMode === 'weight') {
        setWeight(weightCandidates[weightIdx]);
        setInputMode('prompt');
        return;
      }
    }
  });

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold underline>
        Inference — Interactive Chat
      </Text>
      <Box marginY={0} />

      {/* Config line */}
      <Box flexDirection="row" marginY={0} columnGap={2}>
        <Box flexDirection="row">
          <Text bold>Model: </Text>
          <Text color={inputMode === 'model' ? 'yellow' : undefined}>
            {modelType}
          </Text>
        </Box>
        <Box flexDirection="row">
          <Text bold>Weight: </Text>
          <Text color={inputMode === 'weight' ? 'yellow' : undefined}>
            {effectiveWeight || '<none>'}
          </Text>
        </Box>
        <Box flexDirection="row">
          <Text bold>Max tokens: </Text>
          <Text color={inputMode === 'tokens' ? 'yellow' : undefined}>{maxTokens}</Text>
        </Box>
        <Box flexDirection="row">
          <Text bold>Temp: </Text>
          <Text color={inputMode === 'temp' ? 'yellow' : undefined}>{temperature}</Text>
        </Box>
      </Box>

      {/* Available weights for current model type */}
      {weightCandidates.length > 0 ? (
        <Box flexDirection="row" marginY={0}>
          <Text dimColor>
            available weights:{' '}
            {weightCandidates.map((w, i) => (
              <Text key={w} color={w === effectiveWeight ? 'blue' : undefined} bold={w === effectiveWeight}>
                {w === effectiveWeight ? ` [${w}]` : ` ${w}`}
              </Text>
            ))}
          </Text>
        </Box>
      ) : (
        <Box marginY={0}>
          <Text dimColor>No models found. Train or download weights first.</Text>
        </Box>
      )}

      {/* Load / Reload button */}
      <Box marginY={0}>
        <Text>
          <Text color="blue" bold>
            [L]oad model
          </Text>
          <Text dimColor> | </Text>
          <Text color="green" bold>
            [G]enerate
          </Text>
          <Text dimColor> on Enter</Text>
        </Text>
      </Box>

      {/* Status */}
      <Box marginY={0}>
        {session.modelStatus ? (
          <Text color="green">{session.modelStatus}</Text>
        ) : session.error ? (
          <Text color="red">{session.error}</Text>
        ) : null}
      </Box>

      {/* Output area */}
      <Box
        flexDirection="column"
        borderStyle="round"
        borderColor="gray"
        paddingX={1}
        marginY={0}
        height={16}
      >
        <Text bold>Output</Text>
        <Box marginY={0} />
        {session.tokens.length === 0 && !session.generationDone ? (
          <Text dimColor>Type a prompt and press Enter to generate.</Text>
        ) : (
          <Box flexDirection="column" overflowY="hidden" minHeight={0}>
            <Text wrap="wrap">{session.tokens.join('')}</Text>
            {!session.generationDone && session.tokens.length > 0 ? (
              <Text color="yellow">▍</Text>
            ) : null}
          </Box>
        )}
      </Box>

      {/* Input area */}
      <Box flexDirection="column" marginY={0}>
        {inputMode === 'prompt' && (
          <Box flexDirection="row">
            <Text bold color="blue">
              {'> '}
            </Text>
            <Box flexGrow={1}>
              <TextInput
                value={prompt}
                onChange={setPrompt}
                onSubmit={handleSubmit}
                placeholder="Enter your prompt..."
              />
            </Box>
          </Box>
        )}
        {inputMode === 'model' && (
          <Box flexDirection="row">
            <Text bold>Model type: </Text>
            <TextInput
              value={modelType}
              onChange={(v) => {
                setModelType(v as 'lm' | 'vlm' | 'rlm');
                setLoaded(false);
              }}
              onSubmit={() => {
                // Auto-pick first weight for this type
                const first = weightCandidates[0];
                if (first) setWeight(first);
                setInputMode('prompt');
              }}
              placeholder="lm | vlm | rlm"
            />
          </Box>
        )}
        {inputMode === 'weight' && (
          <Box flexDirection="column">
            <Text bold>Select weight (↑↓ enter to confirm):</Text>
            {weightCandidates.map((w, i) => (
              <Text key={w} color={i === weightIdx ? 'blue' : undefined} bold={i === weightIdx}>
                {i === weightIdx ? '▸ ' : '  '}{w}
              </Text>
            ))}
          </Box>
        )}
        {inputMode === 'tokens' && (
          <Box flexDirection="row">
            <Text bold>Max tokens: </Text>
            <TextInput
              value={maxTokens}
              onChange={setMaxTokens}
              onSubmit={() => setInputMode('prompt')}
              placeholder="512"
            />
          </Box>
        )}
        {inputMode === 'temp' && (
          <Box flexDirection="row">
            <Text bold>Temperature: </Text>
            <TextInput
              value={temperature}
              onChange={setTemperature}
              onSubmit={() => setInputMode('prompt')}
              placeholder="0.7"
            />
          </Box>
        )}
      </Box>

      <Box>
        <Text dimColor>
          Tab to switch fields | ↑↓ select weight | L to load model | Enter to generate
        </Text>
      </Box>
    </Box>
  );
}
