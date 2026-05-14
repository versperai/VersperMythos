import React from 'react';
import { Box, Text } from 'ink';
import type { BackendState } from '../hooks/useBackendSession';

export function Dashboard({
  session,
}: {
  session: BackendState & { sendRequest: (req: any) => void };
}): React.JSX.Element {
  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold underline>
        VersperMythos TUI
      </Text>
      <Box marginY={1} />

      <Text bold>Available Models</Text>
      <Box marginY={0} />
      <Box flexDirection="column" paddingX={1}>
        <Text>  • LM (Language Model) — minimind-3</Text>
        <Text>  • VLM (Vision-Language Model) — minimind-3v</Text>
        <Text>  • RLM (Recurrent Language Model)</Text>
      </Box>

      <Box marginY={1} />
      <Text bold>Quick Actions</Text>
      <Box marginY={0} />
      <Box flexDirection="column" paddingX={1}>
        <Text>  [1] Dashboard    [2] Data EDA    [3] Inference</Text>
        <Text>  [4] Training     [5] Benchmark</Text>
      </Box>

      <Box marginY={1} />
      <Text bold>Backend Status</Text>
      <Box marginY={0} />
      <Box flexDirection="column" paddingX={1}>
        <Text>
          Status:{' '}
          {session.ready ? (
            <Text color="green">Connected</Text>
          ) : (
            <Text color="yellow">Connecting...</Text>
          )}
        </Text>
        {session.modelStatus ? <Text>Model: {session.modelStatus}</Text> : null}
        {session.error ? <Text color="red">Error: {session.error}</Text> : null}
      </Box>

      <Box marginY={1} />
      <Text bold>Model Loading</Text>
      <Box marginY={0} />
      <Box flexDirection="column" paddingX={1}>
        <Text>Use Inference tab to load and chat with a model.</Text>
        <Text>Training tab to launch fine-tuning jobs.</Text>
        <Text>Data EDA to explore datasets.</Text>
        <Text>Benchmark to run evaluations.</Text>
      </Box>

      <Box marginY={1} />
      <Text dimColor>Press number keys 1-5 to switch tabs, Esc to exit.</Text>
    </Box>
  );
}
