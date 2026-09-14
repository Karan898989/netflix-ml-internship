/**
 * Netflix ML Internship — Interactive Frontend Application
 * Handles tab navigation, autocomplete, live recommendation queries,
 * classifier inference calls, and Chart.js analytics visual rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initRecommender();
  initClassifier();
  initCharts();
});

/* ==========================================================================
   Tab Navigation
   ========================================================================== */
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.dataset.target;
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      // If switching to analytics, ensure charts resize properly
      if (targetId === 'tab-analytics') {
        window.dispatchEvent(new Event('resize'));
      }
    });
  });
}

/* ==========================================================================
   Task 1: Live Recommendation Engine
   ========================================================================== */
function initRecommender() {
  const searchInput = document.getElementById('search-title');
  const autocompleteList = document.getElementById('autocomplete-list');
  const btnSearch = document.getElementById('btn-get-recs');
  const typeFilter = document.getElementById('filter-type');
  const recsContainer = document.getElementById('recs-container');
  const presetChips = document.querySelectorAll('.preset-chip');

  if (!searchInput) return;

  // Autocomplete search listener
  let debounceTimeout = null;
  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimeout);
    const q = searchInput.value.trim();
    if (q.length < 2) {
      autocompleteList.style.display = 'none';
      return;
    }

    debounceTimeout = setTimeout(async () => {
      try {
        const resp = await fetch(`/api/titles?q=${encodeURIComponent(q)}`);
        const titles = await resp.json();
        if (titles.length > 0) {
          autocompleteList.innerHTML = titles
            .map(t => `<div class="autocomplete-item" data-title="${t}">${t}</div>`)
            .join('');
          autocompleteList.style.display = 'block';

          autocompleteList.querySelectorAll('.autocomplete-item').forEach(item => {
            item.addEventListener('click', () => {
              searchInput.value = item.dataset.title;
              autocompleteList.style.display = 'none';
              fetchRecommendations(item.dataset.title);
            });
          });
        } else {
          autocompleteList.style.display = 'none';
        }
      } catch (err) {
        console.error('Autocomplete error:', err);
      }
    }, 200);
  });

  // Close autocomplete on click outside
  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !autocompleteList.contains(e.target)) {
      autocompleteList.style.display = 'none';
    }
  });

  // Search Button Click
  btnSearch.addEventListener('click', () => {
    const title = searchInput.value.trim();
    if (title) fetchRecommendations(title);
  });

  // Enter key press
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      autocompleteList.style.display = 'none';
      const title = searchInput.value.trim();
      if (title) fetchRecommendations(title);
    }
  });

  // Preset Chips
  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const title = chip.dataset.title;
      searchInput.value = title;
      autocompleteList.style.display = 'none';
      fetchRecommendations(title);
    });
  });

  // Filter change
  typeFilter.addEventListener('change', () => {
    const title = searchInput.value.trim();
    if (title) fetchRecommendations(title);
  });

  // Initial recommendation load with default title
  fetchRecommendations('Stranger Things');
}

async function fetchRecommendations(title) {
  const recsContainer = document.getElementById('recs-container');
  const typeFilter = document.getElementById('filter-type').value;
  const statusMsg = document.getElementById('recs-status-msg');

  statusMsg.innerHTML = `<span style="color: var(--accent-gold);">🔍 Finding top semantic matches for "${title}"...</span>`;
  recsContainer.innerHTML = '';

  try {
    const filterParam = typeFilter !== 'All' ? `&type=${encodeURIComponent(typeFilter)}` : '';
    const resp = await fetch(`/api/recommend?title=${encodeURIComponent(title)}&top_n=6${filterParam}`);
    const data = await resp.json();

    if (data.error || !data.recommendations || data.recommendations.length === 0) {
      statusMsg.innerHTML = `<span style="color: #ff5454;">⚠️ ${data.error || 'No recommendations found.'}</span>`;
      return;
    }

    statusMsg.innerHTML = `Showing <strong>${data.recommendations.length}</strong> recommendations for <strong>"${title}"</strong>:`;

    recsContainer.innerHTML = data.recommendations.map(r => `
      <div class="movie-card">
        <div>
          <div class="card-top">
            <h4 class="card-title">${escapeHtml(r.title)}</h4>
            <span class="match-badge">${r.match_score_pct}% MATCH</span>
          </div>
          <div class="card-meta">
            <span class="meta-pill">${r.type}</span>
            <span class="meta-pill">${r.release_year || 'N/A'}</span>
          </div>
          <div class="card-genres">${escapeHtml(r.listed_in)}</div>
          <p class="card-desc">${escapeHtml(r.description)}</p>
        </div>
        <div class="card-footer">
          <strong>Director:</strong> ${escapeHtml(r.director || 'Unknown')}
        </div>
      </div>
    `).join('');
  } catch (err) {
    statusMsg.innerHTML = `<span style="color: #ff5454;">⚠️ Error connecting to recommendation engine.</span>`;
    console.error(err);
  }
}

/* ==========================================================================
   Task 2: Live Content Type Classifier
   ========================================================================== */
function initClassifier() {
  const btnPredict = document.getElementById('btn-run-predict');
  if (!btnPredict) return;

  btnPredict.addEventListener('click', async () => {
    const title = document.getElementById('input-title').value.trim() || 'Untitled Project';
    const country = document.getElementById('input-country').value;
    const rating = document.getElementById('input-rating').value;
    const releaseYear = document.getElementById('input-year').value;
    const description = document.getElementById('input-desc').value.trim();

    // Collect checked genres
    const genreCheckboxes = document.querySelectorAll('input[name="genre-check"]:checked');
    const genres = Array.from(genreCheckboxes).map(cb => cb.value);

    if (genres.length === 0) {
      alert('Please select at least one genre.');
      return;
    }

    btnPredict.disabled = true;
    btnPredict.innerHTML = '⚡ Running ML Inference...';

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title,
          genres: genres,
          country: country,
          rating: rating,
          release_year: releaseYear,
          description: description
        })
      });

      const data = await resp.json();

      // Update UI elements
      const badge = document.getElementById('pred-result-badge');
      const meterFill = document.getElementById('pred-meter-fill');
      const moviePct = document.getElementById('pred-movie-pct');
      const tvPct = document.getElementById('pred-tv-pct');
      const confText = document.getElementById('pred-confidence-text');

      badge.textContent = data.prediction;
      if (data.prediction === 'Movie') {
        badge.style.background = 'linear-gradient(135deg, #e50914, #ff5454)';
      } else {
        badge.style.background = 'linear-gradient(135deg, #1f77b4, #3a86ff)';
      }

      moviePct.textContent = `${data.movie_probability}%`;
      tvPct.textContent = `${data.tv_show_probability}%`;
      meterFill.style.width = `${data.movie_probability}%`;

      confText.innerHTML = `Model Confidence: <strong>${data.confidence}%</strong> (${data.prediction} Likelihood)`;
    } catch (err) {
      console.error('Classification error:', err);
      alert('Failed to run model prediction.');
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = '⚡ Predict Content Type';
    }
  });
}

/* ==========================================================================
   Task 6: Interactive Chart.js Visual Dashboard
   ========================================================================== */
let chartsInitialized = false;

async function initCharts() {
  if (chartsInitialized) return;
  
  try {
    const resp = await fetch('/api/analytics');
    const data = await resp.json();
    if (!data.yearly || !data.yearly.years) return;

    renderCatalogGrowthChart(data.yearly);
    renderCountryChart(data.countries);
    renderRatingChart(data.ratings);
    renderLongitudinalTrendChart(data.yearly);
    chartsInitialized = true;
  } catch (err) {
    console.error('Failed to load chart analytics:', err);
  }
}

function renderCatalogGrowthChart(yearly) {
  const ctx = document.getElementById('chart-growth');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: yearly.years,
      datasets: [
        {
          label: 'Movies Added',
          data: yearly.movies,
          backgroundColor: '#e50914',
          borderRadius: 4
        },
        {
          label: 'TV Shows Added',
          data: yearly.tv_shows,
          backgroundColor: '#3a86ff',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a3a3a3' } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a3a3a3' } }
      },
      plugins: {
        legend: { labels: { color: '#ffffff' } }
      }
    }
  });
}

function renderCountryChart(countries) {
  const ctx = document.getElementById('chart-countries');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: countries.labels,
      datasets: [{
        label: 'Titles Produced',
        data: countries.counts,
        backgroundColor: '#ffb703',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a3a3a3' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a3a3a3' } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderRatingChart(ratings) {
  const ctx = document.getElementById('chart-ratings');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ratings.labels,
      datasets: [{
        data: ratings.counts,
        backgroundColor: ['#e50914', '#ffb703', '#2ec4b6', '#3a86ff', '#8338ec', '#fb5607'],
        borderColor: '#1e1e1e',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: '#ffffff' } }
      }
    }
  });
}

function renderLongitudinalTrendChart(yearly) {
  const ctx = document.getElementById('chart-trends');
  if (!ctx) return;

  const totalPerYear = yearly.years.map((_, i) => yearly.movies[i] + yearly.tv_shows[i]);
  const movieShare = yearly.years.map((_, i) => totalPerYear[i] ? ((yearly.movies[i] / totalPerYear[i]) * 100).toFixed(1) : 0);
  const tvShare = yearly.years.map((_, i) => totalPerYear[i] ? ((yearly.tv_shows[i] / totalPerYear[i]) * 100).toFixed(1) : 0);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: yearly.years,
      datasets: [
        {
          label: 'Movie Share (%)',
          data: movieShare,
          borderColor: '#e50914',
          backgroundColor: 'rgba(229, 9, 20, 0.1)',
          fill: true,
          tension: 0.3
        },
        {
          label: 'TV Show Share (%)',
          data: tvShare,
          borderColor: '#3a86ff',
          backgroundColor: 'rgba(58, 134, 255, 0.1)',
          fill: true,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a3a3a3' } },
        y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a3a3a3' } }
      },
      plugins: {
        legend: { labels: { color: '#ffffff' } }
      }
    }
  });
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
