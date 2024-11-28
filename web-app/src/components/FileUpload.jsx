import React from 'react';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { toast } from 'react-toastify';

const FileUploadSchema = Yup.object().shape({
  csvFile: Yup.mixed()
    .required('Por favor selecciona un archivo')
    .test('fileType', 'Solo se permiten archivos CSV', (value) => {
      return value && value.name.toLowerCase().endsWith('.csv');
    })
    .test('fileSize', 'El archivo no debe exceder 10 MB', (value) => {
      return value && value.size <= 10 * 1024 * 1024; // 10 MB
    })
});

const FileUpload = () => {
  const handleFileSubmit = (values, { setSubmitting, resetForm }) => {
    const formData = new FormData();
    formData.append('csvFile', values.csvFile);

    // Simular envío de archivo
    try {
      // Aquí iría la lógica real de envío, por ejemplo con fetch
      setTimeout(() => {
        toast.success('Archivo CSV cargado exitosamente', {
          position: "top-right",
          autoClose: 3000,
        });
        setSubmitting(false);
        resetForm();
      }, 1500);
    } catch (error) {
      toast.error('Error al cargar el archivo', {
        position: "top-right",
        autoClose: 3000,
      });
      setSubmitting(false);
    }
  };

  return (
    <div className="p-8 max-w-xl mx-auto bg-white shadow-md rounded-lg mt-10">
      <h2 className="text-2xl font-bold mb-6 text-center">Cargar Archivo CSV</h2>
      
      <Formik
        initialValues={{ csvFile: null }}
        validationSchema={FileUploadSchema}
        onSubmit={handleFileSubmit}
      >
        {({ errors, touched, setFieldValue, isSubmitting }) => (
          <Form>
            <div className="mb-4">
              <label 
                htmlFor="csvFile" 
                className="block text-gray-700 text-sm font-bold mb-2"
              >
                Seleccionar Archivo CSV
              </label>
              <input
                id="csvFile"
                name="csvFile"
                type="file"
                accept=".csv"
                onChange={(event) => {
                  setFieldValue('csvFile', event.currentTarget.files[0]);
                }}
                className="w-full p-2 border rounded-md"
              />
              {errors.csvFile && touched.csvFile && (
                <div className="text-red-500 text-sm mt-1">
                  {errors.csvFile}
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className={`
                w-full p-2 rounded-md text-white 
                ${isSubmitting 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600'
                }
              `}
            >
              {isSubmitting ? 'Cargando...' : 'Cargar Archivo'}
            </button>
          </Form>
        )}
      </Formik>
    </div>
  );
};

export default FileUpload;