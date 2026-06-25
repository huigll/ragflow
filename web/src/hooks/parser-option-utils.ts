export type ParserOption = {
  value: string;
  label: string;
};

export function normalizeParserOptions(
  parserIds: string | null | undefined,
  defaultParsers: ParserOption[],
): ParserOption[] {
  const parserArray = (parserIds ?? '')
    .split(',')
    .map((parser) => parser.trim())
    .filter(Boolean);

  if (parserArray.length === 0) {
    return defaultParsers;
  }

  const defaultLabelByValue = new Map(
    defaultParsers.map((parser) => [parser.value, parser.label]),
  );

  return parserArray
    .map((parser) => {
      const [rawValue, ...rawLabelParts] = parser.split(':');
      const value = rawValue.trim();
      const label =
        rawLabelParts.join(':').trim() ||
        defaultLabelByValue.get(value) ||
        value;

      return { value, label };
    })
    .filter((parser) => parser.value);
}
