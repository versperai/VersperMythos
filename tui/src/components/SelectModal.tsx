import React, { useCallback, useMemo, useState } from 'react';
import { Box, Text, useInput } from 'ink';

export type SelectOption = {
  value: string;
  label: string;
  description?: string;
};

export type SelectGroup = {
  title: string;
  items: SelectOption[];
};

type Props = {
  groups: SelectGroup[];
  onSelect: (value: string) => void;
  onCancel: () => void;
  title?: string;
};

/** Flatten groups into a single-indexed list of { groupIndex, itemIndex } */
function buildFlatIndex(groups: SelectGroup[]) {
  const flat: Array<{ groupIdx: number; itemIdx: number }> = [];
  for (let g = 0; g < groups.length; g++) {
    for (let i = 0; i < groups[g].items.length; i++) {
      flat.push({ groupIdx: g, itemIdx: i });
    }
  }
  return flat;
}

export function SelectModal({ groups, onSelect, onCancel, title }: Props): React.JSX.Element {
  const flat = useMemo(() => buildFlatIndex(groups), [groups]);
  const [cursor, setCursor] = useState(0);

  const selected = flat[cursor];

  useInput((_input, key) => {
    if (key.return && selected) {
      onSelect(groups[selected.groupIdx].items[selected.itemIdx].value);
      return;
    }
    if (key.escape) {
      onCancel();
      return;
    }
    if (key.upArrow) {
      setCursor((c) => (c <= 0 ? flat.length - 1 : c - 1));
      return;
    }
    if (key.downArrow) {
      setCursor((c) => (c >= flat.length - 1 ? 0 : c + 1));
      return;
    }
  });

  // Height constraint: max visible rows
  const MAX_VISIBLE = process.stdout.rows ? process.stdout.rows - 8 : 20;
  const scrollOffset = Math.max(0, cursor - Math.floor(MAX_VISIBLE / 2));

  let row = 0;
  let visible = 0;

  const renderGroupContent = (g: SelectGroup, gIdx: number): React.JSX.Element | null => {
    if (g.items.length === 0) return null;
    const rows: React.JSX.Element[] = [];

    // Group header
    rows.push(
      <Text key={`h-${gIdx}`} bold color="cyan">
        {'  '}╶ {g.title}
      </Text>,
    );
    row++;

    for (let i = 0; i < g.items.length; i++) {
      const globalIdx = flat.findIndex((f) => f.groupIdx === gIdx && f.itemIdx === i);
      const isSelected = globalIdx === cursor;
      const isVisible = visible >= scrollOffset && visible < scrollOffset + MAX_VISIBLE;

      if (isVisible) {
        rows.push(
          <Box key={`i-${gIdx}-${i}`} flexDirection="row">
            <Text color={isSelected ? 'blue' : undefined} bold={isSelected}>
              {isSelected ? '  ▸ ' : '    '}
              {g.items[i].label}
            </Text>
            {g.items[i].description ? (
              <Text dimColor>  {g.items[i].description}</Text>
            ) : null}
          </Box>,
        );
      }
      row++;
      visible++;
    }

    return <>{rows}</>;
  };

  return (
    <Box flexDirection="column" paddingX={1} paddingY={1}>
      <Box borderStyle="round" borderColor="blue" flexDirection="column" paddingX={1}>
        {title ? (
          <Text bold underline>
            {title}
          </Text>
        ) : null}
        <Box flexDirection="column" marginTop={0}>
          {groups.map((g, gIdx) => (
            <Box key={`group-${gIdx}`} flexDirection="column">
              {renderGroupContent(g, gIdx)}
            </Box>
          ))}
        </Box>
        <Box marginTop={0}>
          <Text dimColor>
            {'\u2191\u2193'} navigate · Enter select · Esc cancel
          </Text>
        </Box>
      </Box>
    </Box>
  );
}
