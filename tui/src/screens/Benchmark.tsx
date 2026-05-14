import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';

import type { BackendState } from '../hooks/useBackendSession';

export function Benchmark({
  session,
}: {
  session: BackendState & { sendRequest: (req: any) => void };
}): React.JSX.Element {
  const [modelType, setModelType] = useState('lm');
  const [weight, setWeight] = useState('minimind');
  const [benchName, setBenchName] = useState('perplexity');

  const handleRun = (): void => {
    session.sendRequest({
      type: 'bench_run',
      model_type: modelType as any,
      weight,
      bench_name: benchName,
    });
  };

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold underline>
        Benchmark — Evaluation Suite
      </Text>
      <Box marginY={0} />

      <Box flexDirection="column" marginY={0}>
        <Box flexDirection="row">
          <Text bold>Model type: </Text>
          <TextInput
            value={modelType}
            onChange={setModelType}
            placeholder="lm | vlm | rlm"
          />
        </Box>
        <Box flexDirection="row">
          <Text bold>Weight: </Text>
          <TextInput value={weight} onChange={setWeight} placeholder="minimind" />
        </Box>
        <Box flexDirection="row">
          <Text bold>Benchmark: </Text>
          <TextInput
            value={benchName}
            onChange={setBenchName}
            placeholder="perplexity | accuracy"
          />
        </Box>
      </Box>

      <Box marginY={0}>
        <Text color="blue" bold>
          {'> '}Press Enter to run benchmark
        </Text>
      </Box>

      {/* Results */}
      <Box
        flexDirection="column"
        flexGrow={1}
        borderStyle="round"
        borderColor="gray"
        paddingX={1}
        marginY={0}
      >
        <Text bold>Results</Text>
        <Box marginY={0} />
        {session.benchResult ? (
          <Box flexDirection="column">
            {Object.entries(session.benchResult).map(([k, v]) => (
              <Text key={k}>
                {k}: {String(v ?? '')}
              </Text>
            ))}
          </Box>
        ) : (
          <Text dimColor>No benchmark results yet. Configure and run a benchmark.</Text>
        )}
      </Box>

      <Box>
        <Text dimColor>
          Configure benchmark parameters and press Enter to run.
        </Text>
      </Box>
    </Box>
  );
}
