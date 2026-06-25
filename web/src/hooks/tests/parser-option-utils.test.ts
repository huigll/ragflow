import {
  normalizeParserOptions,
  type ParserOption,
} from '../parser-option-utils';

const defaultParsers: ParserOption[] = [
  { value: 'naive', label: 'General' },
  { value: 'qa', label: 'Q&A' },
  { value: 'table', label: 'Table' },
];

describe('normalizeParserOptions', () => {
  it('uses the default parser list when tenant parser ids are empty', () => {
    expect(normalizeParserOptions('', defaultParsers)).toEqual(defaultParsers);
  });

  it('keeps labels supplied by tenant parser ids', () => {
    expect(
      normalizeParserOptions(
        'naive:Custom general,qa:Custom QA',
        defaultParsers,
      ),
    ).toEqual([
      { value: 'naive', label: 'Custom general' },
      { value: 'qa', label: 'Custom QA' },
    ]);
  });

  it('falls back to default labels when tenant parser ids only contain values', () => {
    expect(normalizeParserOptions('naive,qa,table', defaultParsers)).toEqual([
      { value: 'naive', label: 'General' },
      { value: 'qa', label: 'Q&A' },
      { value: 'table', label: 'Table' },
    ]);
  });

  it('uses the parser value as a final label fallback', () => {
    expect(normalizeParserOptions('custom_parser', defaultParsers)).toEqual([
      { value: 'custom_parser', label: 'custom_parser' },
    ]);
  });
});
