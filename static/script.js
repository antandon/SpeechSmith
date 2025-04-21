function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({
        behavior: 'smooth'
    });
}

// Form submission handler
document.getElementById('contactForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const comments = document.getElementById('comments').value;

    const csvData = [
        ['Name', 'Email', 'Comments'],
        [name, email, comments]
    ].map(row => row.map(field => `"${field}"`).join(',')).join('\n');

    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('href', url);
    a.setAttribute('download', 'contact_form_data.csv');

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    const successMessage = document.getElementById('successMessage');
    successMessage.style.display = 'block';

    this.reset();

    setTimeout(() => {
        successMessage.style.display = 'none';
    }, 3000);
});

// Limit checkbox selection to 2
const checkboxes = document.querySelectorAll('input[type="checkbox"]');
let checkedCount = 0;

checkboxes.forEach(checkbox => {
    checkbox.addEventListener('change', function() {
        if (this.checked) {
            checkedCount++;
            if (checkedCount > 2) {
                this.checked = false;
                checkedCount--;
                alert('You can only select a maximum of 2 options');
            }
        } else {
            checkedCount--;
        }
    });
});

// // Page navigation
// const formPage = document.getElementById('formPage');
// const navPage = document.getElementById('navPage');

// function showNavPage() {
//     formPage.style.display = 'none';
//     navPage.style.display = 'block';
// }

// function showFormPage() {
//     navPage.style.display = 'none';
//     formPage.style.display = 'block';
// }