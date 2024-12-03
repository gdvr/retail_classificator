# Usar una imagen base de Node con una versión compatible
FROM node:20-alpine3.19

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar package.json y package-lock.json
COPY package*.json ./

# Instalar dependencias
RUN npm install

# Copiar todo el código fuente
COPY . .

# Construir la aplicación para producción
RUN npm run build

# Instalar un servidor web ligero para servir archivos estáticos
RUN npm install -g serve

# Exponer el puerto 8080
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["serve", "-s", "dist", "-l", "8080"]