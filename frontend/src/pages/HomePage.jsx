import { Link } from "react-router-dom";

function HomePage() {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
            {/* Header */}
            <header className="w-full max-w-4xl flex justify-between items-center py-6">
                <h1 className="text-4xl font-bold text-gray-900">GradeMate</h1>
                <nav>
                    <ul className="flex space-x-6">
                        <li><Link to="/upload" className="text-lg text-gray-700 hover:text-blue-500">Upload Exam</Link></li>
                        <li><Link to="/results" className="text-lg text-gray-700 hover:text-blue-500">Results</Link></li>
                        <li><Link to="/settings" className="text-lg text-gray-700 hover:text-blue-500">Settings</Link></li>
                    </ul>
                </nav>
            </header>

            {/* Hero Section */}
            <section className="text-center mt-16">
                <h2 className="text-5xl font-extrabold text-gray-900">Automated Exam Grading System</h2>
                <p className="mt-4 text-lg text-gray-600">Grade exams faster and more efficiently with AI-powered grading.</p>
                <Link to="/upload" className="mt-6 inline-block bg-blue-500 text-white text-lg px-6 py-3 rounded-lg shadow-md hover:bg-blue-600">Get Started</Link>
            </section>

            {/* Social Media Links */}
            <footer className="mt-16 text-center">
                <p className="text-gray-600">Follow us:</p>
                <div className="flex space-x-4 mt-2">
                    <a href="#" className="text-gray-700 hover:text-blue-500">Twitter</a>
                    <a href="#" className="text-gray-700 hover:text-blue-500">LinkedIn</a>
                    <a href="#" className="text-gray-700 hover:text-blue-500">GitHub</a>
                </div>
            </footer>
        </div>
    );
}

export default HomePage;
