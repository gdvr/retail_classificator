import React, { useState } from 'react';
import {
    BrowserRouter as Router,
    Routes,
    Route,
} from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Sidebar from './Sidebar';
import HomeApp from './HomeApp';
// import FileTable from './FileTable';
import FileUpload from './FileUpload';

const Dashboard = () => {
    const [sidebarOpen, setSidebarOpen] = useState(true);

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    return (
        <Router>
            <div className="flex h-screen">
                <Sidebar isOpen={sidebarOpen} onToggle={toggleSidebar} />

                <main
                    className={`
              flex-1 transition-all duration-300
              ${sidebarOpen ? 'ml-64' : 'ml-20'}
            `}
                >
                    <Routes>
                        <Route path="/" element={<HomeApp />} />
                        <Route path="/archivos" element={<FileUpload />} />
                    </Routes>
                </main>

                <ToastContainer />
            </div>
        </Router>
    );
};

export default Dashboard;