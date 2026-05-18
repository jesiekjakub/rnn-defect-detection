import { onBeforeUnmount, onMounted, ref, watch, type Ref } from 'vue';
import Plotly, { type Data, type Layout, type Config } from 'plotly.js-dist-min';

const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {
    family: 'Inter, system-ui, sans-serif',
    color: '#cbd5f5',
    size: 12,
  },
  margin: { l: 48, r: 16, t: 28, b: 36 },
  xaxis: {
    gridcolor: 'rgba(148,163,184,0.08)',
    zerolinecolor: 'rgba(148,163,184,0.12)',
    tickfont: { size: 11 },
  },
  yaxis: {
    gridcolor: 'rgba(148,163,184,0.08)',
    zerolinecolor: 'rgba(148,163,184,0.12)',
    tickfont: { size: 11 },
  },
  legend: {
    orientation: 'h',
    x: 0,
    y: -0.18,
    bgcolor: 'rgba(0,0,0,0)',
    font: { size: 11 },
  },
};

const BASE_CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: [
    'lasso2d',
    'select2d',
    'autoScale2d',
    'hoverClosestCartesian',
    'hoverCompareCartesian',
    'toggleSpikelines',
  ],
};

export function mergeLayout(layout: Partial<Layout>): Partial<Layout> {
  return {
    ...BASE_LAYOUT,
    ...layout,
    xaxis: { ...BASE_LAYOUT.xaxis, ...(layout.xaxis ?? {}) },
    yaxis: { ...BASE_LAYOUT.yaxis, ...(layout.yaxis ?? {}) },
    legend: { ...BASE_LAYOUT.legend, ...(layout.legend ?? {}) },
    margin: { ...BASE_LAYOUT.margin, ...(layout.margin ?? {}) },
  };
}

export function usePlotly(
  data: Ref<Data[]>,
  layout: Ref<Partial<Layout>>,
  config: Partial<Config> = {},
) {
  const container = ref<HTMLDivElement | null>(null);

  const draw = async () => {
    if (!container.value) return;
    await Plotly.react(container.value, data.value, mergeLayout(layout.value), {
      ...BASE_CONFIG,
      ...config,
    });
  };

  onMounted(draw);
  watch([data, layout], draw, { deep: true });
  onBeforeUnmount(() => {
    if (container.value) Plotly.purge(container.value);
  });

  return { container };
}
