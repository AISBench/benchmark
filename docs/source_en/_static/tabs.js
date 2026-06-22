document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.tab-container').forEach(function(container) {
        var buttons = container.querySelectorAll('.tab-btn');
        var panels = container.querySelectorAll('.tab-panel');

        function activateTab(btn) {
            var target = btn.getAttribute('data-tab');
            buttons.forEach(function(b) { b.classList.remove('active'); });
            panels.forEach(function(p) { p.classList.remove('active'); });
            btn.classList.add('active');
            var panel = container.querySelector('.tab-panel[data-tab="' + target + '"]');
            if (panel) panel.classList.add('active');
        }

        buttons.forEach(function(btn) {
            btn.addEventListener('click', function() {
                activateTab(this);
            });
        });

        var activeBtn = container.querySelector('.tab-btn.active');
        if (activeBtn) {
            activateTab(activeBtn);
        } else if (buttons.length > 0) {
            activateTab(buttons[0]);
        }
    });
});
