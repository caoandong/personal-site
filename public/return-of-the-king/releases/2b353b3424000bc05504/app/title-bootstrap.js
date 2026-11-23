
    (function () {
      var q = new URLSearchParams(window.location.search);
      var wantTitle = (q.get('film') !== '1' || q.get('title') === '1' || q.get('title') === 'true') && q.get('mode') !== 'play' && q.get('play') !== '1';
      if (wantTitle) document.body.classList.add('title-screen');
    })();
  