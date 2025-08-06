import { createChart } from 'lightweight-charts';

const chart = createChart(document.createElement('div'), {
  width: 400,
  height: 300,
});

// Try to see what methods are available
console.log('Chart methods:', Object.getOwnPropertyNames(chart));
console.log('Available series methods:', Object.getOwnPropertyNames(chart.__proto__));
