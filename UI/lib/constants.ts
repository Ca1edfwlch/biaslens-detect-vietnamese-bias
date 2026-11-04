// Display names for the 14-category severity detector (src/training/
// inference_llm_lora.py). Mirrors the category rubric in
// src/training/inference_severity_llm.py's CATEGORY_DEFINITIONS -- keep the
// key set in sync if that rubric changes.
export const CATEGORY_LABELS: Record<string, string> = {
  sexism: 'Sexism',
  lgbtq_bias: 'LGBTQ+ Bias',
  ethnic_minority_bias: 'Ethnic Minority Bias',
  regional_bias: 'Regional Bias',
  religion: 'Religion',
  ageism: 'Ageism',
  class_poverty_bias: 'Class / Poverty Bias',
  disability_health_bias: 'Disability / Health Bias',
  ideological_bias: 'Ideological Bias',
  appearance_body_shaming: 'Appearance / Body Shaming',
  linguistic_hierarchical_bias: 'Linguistic Hierarchy Bias',
  educational_cognitive_hierarchy: 'Educational / Cognitive Hierarchy',
  xenophobia: 'Xenophobia',
  moral_lifestyle_bias: 'Moral / Lifestyle Bias',
} as const

export function categoryLabel(category: string): string {
  return CATEGORY_LABELS[category] ?? category
}

// Short (one-word-ish) labels for chart axes, where CATEGORY_LABELS' full
// names ("Educational / Cognitive Hierarchy") would collide with each other
// around a 14-spoke radar chart. Full names still show in tooltips/legends.
export const CATEGORY_SHORT_LABELS: Record<string, string> = {
  sexism: 'Sexism',
  lgbtq_bias: 'LGBTQ+',
  ethnic_minority_bias: 'Ethnicity',
  regional_bias: 'Region',
  religion: 'Religion',
  ageism: 'Age',
  class_poverty_bias: 'Class',
  disability_health_bias: 'Disability',
  ideological_bias: 'Ideology',
  appearance_body_shaming: 'Appearance',
  linguistic_hierarchical_bias: 'Language',
  educational_cognitive_hierarchy: 'Education',
  xenophobia: 'Xenophobia',
  moral_lifestyle_bias: 'Lifestyle',
} as const

export function categoryShortLabel(category: string): string {
  return CATEGORY_SHORT_LABELS[category] ?? categoryLabel(category)
}
