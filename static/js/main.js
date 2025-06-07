/**
 * Patent Database Search - Main JavaScript
 * Contains common functionality for the patent search application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize common UI elements
    initTheme();
    initTooltips();
    
    // Add smooth scrolling for all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });
});

/**
 * Initialize theme-related functionality
 */
function initTheme() {
    // Check if user prefers dark mode
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // We're using dark theme by default, but this could be expanded
    // to allow switching between light/dark themes
    document.body.classList.add('dark-theme');
    
    // Add scroll listener for header shadow
    window.addEventListener('scroll', function() {
        const header = document.querySelector('header');
        if (header) {
            if (window.scrollY > 10) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        }
    });
}

/**
 * Initialize tooltip functionality for elements with data-tooltip attribute
 */
function initTooltips() {
    document.querySelectorAll('[data-tooltip]').forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            
            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = tooltipText;
            
            // Add tooltip to body
            document.body.appendChild(tooltip);
            
            // Position tooltip
            const rect = this.getBoundingClientRect();
            tooltip.style.top = rect.bottom + 10 + 'px';
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            
            // Add show class
            setTimeout(() => tooltip.classList.add('show'), 10);
            
            // Store tooltip reference in element
            this.tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', function() {
            if (this.tooltip) {
                this.tooltip.classList.remove('show');
                
                // Remove tooltip after animation
                setTimeout(() => {
                    if (this.tooltip && this.tooltip.parentNode) {
                        this.tooltip.parentNode.removeChild(this.tooltip);
                    }
                    this.tooltip = null;
                }, 200);
            }
        });
    });
}

/**
 * Format a date string
 * @param {string} dateStr - The date string to format
 * @returns {string} - Formatted date string
 */
function formatDate(dateStr) {
    if (!dateStr) return '';
    
    const date = new Date(dateStr);
    
    // Check if date is valid
    if (isNaN(date.getTime())) return dateStr;
    
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Truncate text to a specified length
 * @param {string} text - The text to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} - Truncated text with ellipsis if needed
 */
function truncateText(text, maxLength = 100) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

/**
 * Highlight search terms in text
 * @param {string} text - The text to highlight terms in
 * @param {string} searchTerm - The search term to highlight
 * @returns {string} - Text with highlighted search terms
 */
function highlightText(text, searchTerm) {
    if (!text || !searchTerm) return text;
    
    // Escape special regex characters in the search term
    const escapedSearchTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    
    // Create a regex that matches the search term with word boundaries
    const regex = new RegExp(`\\b(${escapedSearchTerm})\\b`, 'gi');
    
    // Replace matches with highlighted version
    return text.replace(regex, '<mark>$1</mark>');
} 